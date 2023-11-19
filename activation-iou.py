from glob import glob
from natsort import natsorted
import math

import torch
import numpy as np

# from umap import UMAP
from torchvision.utils import make_grid
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib import colormaps

import data_utils


plt.style.use("ggplot")
plt.style.use("seaborn-v0_8-colorblind")


def get_cache_activation_hook(
    cache_dict={}, key="test_key", mode=None, out_device="cpu"
):
    def hook(module, input0, output):
        if len(output.shape) == 4:  # CNN layers
            if mode is None or mode in ["none", "None", "raw"]:
                pass  # keep output as is
            elif mode == "avg":
                output = output.mean(dim=[2, 3])
            elif mode == "max":
                output = output.amax(dim=[2, 3])
        elif len(output.shape) == 3:  # ViT
            output = output[:, 0].clone()
        elif len(output.shape) == 2:  # FC layers
            pass  # keep output as is
        cache_dict[key] = output.to(out_device)

    return hook


def get_activations(
    target_model,
    images,
    target_layers=["layer4"],
    device="cuda",
    out_device="cpu",
    pool_mode=None,  # 'avg', 'sum' or None
):
    all_features = {target_layer: None for target_layer in target_layers}
    hooks = {}
    for target_layer in target_layers:
        layer = eval(f"target_model.{target_layer}")
        hook = layer.register_forward_hook(
            get_cache_activation_hook(
                cache_dict=all_features,
                key=target_layer,
                mode=pool_mode,
                out_device=out_device,
            )
        )
        hooks[target_layer] = hook
    # Forward
    with torch.no_grad():
        target_model(images.to(device))
    # Remove hooks
    for hook in hooks.values():
        hook.remove()
    # free memory
    #     del all_features
    torch.cuda.empty_cache()
    return all_features


# get model, dataset and loader
device = "cuda"
target_model, target_preprocess = data_utils.get_target_model("resnet50", device)
dataset = data_utils.get_data("imagenet_val", preprocess=target_preprocess)
loader = DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True)

# dataset subset
subset = list(range(0, len(dataset), 10))
sub_dataset = torch.utils.data.Subset(dataset, subset)
sub_loader = DataLoader(sub_dataset, batch_size=256, num_workers=8, pin_memory=True)

target_layers = ["conv1", "layer1", "layer2", "layer3", "layer4"]


# Estimate quantile
quantile = 0.95
quantile_samples = {layer_name: [] for layer_name in target_layers}

print("Estimating quantiles...")
for images, labels in tqdm(sub_loader):
    # grab mini-batch activations for all target layers in a dictionary
    features = get_activations(
        target_model,
        images.to(device),
        target_layers=target_layers,
    )

    # gien a minibatch, for each layer, estimate channel-wise activation quantile
    for layer_name, act in features.items():
        if len(act.shape) == 4:  # Conv layer
            # channel first, then combine all remaining (spatial, and instance) dimensions
            act1 = act.permute(1, 0, 2, 3).reshape(act.shape[1], -1)
        elif len(act.shape) == 2:  # fc layer
            # neuron first
            act1 = act.permute(1, 0)
        q = np.quantile(act1.numpy(), q=quantile, axis=1)
        quantile_samples[layer_name].append(q)

for layer_name, quantile_sample in quantile_samples.items():
    quantile_samples[layer_name] = np.stack(quantile_sample)

# On each layer, take mean of quantile samples as final quantile estimate,
quantiles = {}
for layer_name, quantile_sample in quantile_samples.items():
    quantiles[layer_name] = quantile_sample.mean(axis=0)


# ## Cross-layer neuron-to-neuron IoUs
# Compute neuron-to-neuron IoUs FOR EVERY PAIRS of layers
sum_intersections = {}
sum_unions = {}
ious = {}
print("Aggregating neuron-to-neuron IoUs...")
for images, labels in tqdm(sub_loader):
    features = get_activations(
        target_model,
        images.to(device),
        target_layers=target_layers,
    )

    # take every pairs of layers
    for i, layer_name1 in enumerate(target_layers):
        for layer_name2 in target_layers[i:]:
            act1 = features[layer_name1]
            act2 = features[layer_name2]

            # upsample act2, if act2 smaller than act1
            if act1.shape[2] < act2.shape[2]:
                upsample = torch.nn.Upsample(size=act2.shape[2])
                act1 = upsample(act1)
            elif act1.shape[2] > act2.shape[2]:
                upsample = torch.nn.Upsample(size=act1.shape[2])
                act2 = upsample(act2)
            print(layer_name1, layer_name2, act1.shape, act2.shape)

            n_channel1 = act1.shape[1]
            threshold1 = quantiles[layer_name1]
            threshold1 = torch.from_numpy(threshold1).view(1, threshold1.shape[0], 1, 1)
            act_mask1 = (act1 > threshold1).float()
            channel_first_act_mask1 = act_mask1.permute(1, 0, 2, 3).reshape(
                n_channel1, -1
            )

            n_channel2 = act2.shape[1]
            threshold2 = quantiles[layer_name2]
            threshold2 = torch.from_numpy(threshold2).view(1, threshold2.shape[0], 1, 1)
            act_mask2 = (act2 > threshold2).float()
            channel_first_act_mask2 = act_mask2.permute(1, 0, 2, 3).reshape(
                n_channel2, -1
            )

            key = f"{layer_name1},{layer_name2}"
            key_inv = f"{layer_name2},{layer_name1}"
            if key not in sum_intersections:
                sum_intersections[key] = torch.zeros(n_channel1, n_channel2)
            if key not in sum_unions:
                sum_unions[key] = torch.zeros(n_channel1, n_channel2)
            if key_inv not in sum_intersections:
                sum_intersections[key_inv] = torch.zeros(n_channel2, n_channel1)
            if key_inv not in sum_unions:
                sum_unions[key_inv] = torch.zeros(n_channel2, n_channel1)

            # (sum of) intersections
            sum_intersections[key] += (
                channel_first_act_mask1 @ channel_first_act_mask2.t()
            )
            if key != key_inv:  # different layer
                sum_intersections[key_inv] = sum_intersections[key].t()

            # (sum of) unions
            # for each channel pair, grab all activation masks and compute union

            if layer_name1 == layer_name2:  # same layer, result will be symmetric
                for i in tqdm(range(n_channel1)):
                    for j in range(i + 1, n_channel2):
                        ci = channel_first_act_mask1[i]
                        cj = channel_first_act_mask2[j]
                        union_ij_batch = ((ci + cj) > 0).sum()
                        sum_unions[key][i, j] += union_ij_batch
                        # same layer, matrix is symmetric
                        sum_unions[key][j, i] += union_ij_batch

            else:  # different layers
                for i in tqdm(range(n_channel1)):
                    for j in range(n_channel2):
                        ci = channel_first_act_mask1[i]
                        cj = channel_first_act_mask2[j]
                        union_ij_batch = ((ci + cj) > 0).sum()
                        sum_unions[key][i, j] += union_ij_batch
                        sum_unions[key_inv][j, i] += union_ij_batch

ious = {
    key: sum_intersections[key] / sum_unions[key] for key in sum_intersections.keys()
}


torch.save(
    dict(
        sum_intersections=sum_intersections,
        sum_unions=sum_unions,
        ious=ious,
    ),
    "my_data/iou.pt",
)
