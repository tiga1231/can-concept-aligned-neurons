import os
from tqdm.auto import tqdm

import numpy as np
import torch
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.style.use("seaborn-v0_8-colorblind")

# (https://github.com/fra31/auto-attack/blob/master/autoattack/autoattack.py)
import data_utils
from autoattack import AutoAttack


# load model and defaalt image transform
model, preprocess0 = data_utils.get_target_model(
    target_name="resnet50", device="cuda", weights="default"
)


# transforms
def denormalize(x):
    std = torch.tensor(preprocess0.std).view(3, 1, 1)
    mean = torch.tensor(preprocess0.mean).view(3, 1, 1)
    x = x * std + mean  # de-normalize
    return x


preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)
normalize = transforms.Normalize(mean=preprocess0.mean, std=preprocess0.std)
to_pil = transforms.ToPILImage()


# model forward pass
def forward_pass(img):
    img = normalize(img)
    return model(img)


# data, loader
dataset = data_utils.get_data("imagenet_val", preprocess)

# create output directory
out_dir = "/home/lim38/dataset/imagenet-val-attack"
os.makedirs(out_dir, exist_ok=True)

# run attacks
adversary = AutoAttack(
    forward_pass,
    norm="Linf",
    eps=15 / 255,  # bound over image domain [0,1]
    version="custom",
    attacks_to_run=["apgd-ce"],
    verbose=True,
)

loader = DataLoader(
    dataset,
    batch_size=100,
    shuffle=False,
    num_workers=16,
)
i = 0  # global image count
for b, [img, target] in enumerate(tqdm(loader)):
    # if b < 457:  # skip previously done jobs
    #     i += target.shape[0]
    # continue
    img, target = img.cuda(), target.cuda()
    adv = adversary.run_standard_evaluation(img, target)
    print()

    for a in adv:
        pil = to_pil(a.detach().cpu())
        subdir, fn = dataset.imgs[i][0].split("/")[-2:]
        os.makedirs(f"{out_dir}/{subdir}", exist_ok=True)
        pil.save(f"{out_dir}/{subdir}/{fn}")
        i += 1
