import utils
import data_utils
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Using CPU...")

# config
dataset_name = "imagenet_val"
target_model_name = "resnet50"
target_layers = [
    "conv1",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
]
pool_mode = "raw"
save_name_format = f"saved_activations/{pool_mode}" + "-{}-{}.pt"

target_model, target_preprocess = data_utils.get_target_model(target_model_name, device)
dataset = data_utils.get_data(dataset_name, target_preprocess)

# target_save_name, clip_save_name, text_save_name  = get_save_names(
#     clip_name=clip_name,
#     target_name=target_name,
#     target_layer="{}",
#     d_probe=d_probe,
#     concept_set=concept_set,
#     pool_mode=pool_mode,
#     save_dir=save_dir,
# )


subset = list(range(0, len(dataset), 10))
dataset = torch.utils.data.Subset(dataset, subset)
utils.save_target_activations(
    target_model,
    dataset,
    save_name_format=save_name_format,
    target_layers=target_layers,
    batch_size=256,
    device=device,
    pool_mode=pool_mode,
)
