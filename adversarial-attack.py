import os
from tqdm.auto import tqdm

import numpy as np
import torch
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim

import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.style.use("seaborn-v0_8-colorblind")


import data_utils
from autoattack import AutoAttack

# (https://github.com/fra31/auto-attack/blob/master/autoattack/autoattack.py)


##load model and defaalt image transform
model, preprocess0 = data_utils.get_target_model(
    target_name="resnet50", device="cuda", weights="default"
)


##transforms
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

normalize = transforms.Normalize(mean=preprocess0.mean, std=preprocess0.std)


def denormalize(x):
    std = torch.tensor(preprocess0.std).view(3, 1, 1)
    mean = torch.tensor(preprocess0.mean).view(3, 1, 1)
    x = x * std + mean  # de-normalize
    return x


to_pil = transforms.ToPILImage()


##data, loader
dataset = data_utils.get_data("imagenet_val", preprocess)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)


## model forward pass
def forward_pass(img):
    img = normalize(img)

    x = img
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    return x


## disable model training
for param in model.parameters():
    param.requires_grad_(False)


def mute_neurons(x, forward_pass, n_iter=40, vis=False, progress=True):
    x = x.clone().requires_grad_(True)
    optimizer = optim.SGD([x], lr=0.01)

    if progress:
        pbar = tqdm(range(n_iter))
    else:
        pbar = range(n_iter)
    for i in pbar:
        x.requires_grad_(True)
        out = forward_pass(x)
        loss = out.pow(2).sum()

        optimizer.zero_grad()
        loss.backward()
        x.grad.data = x.grad.data.sign()  # fast gradient sign
        optimizer.step()

        x.detach_()
        x.clamp_(0, 1)
        if progress:
            pbar.set_postfix({"loss": loss.item()})

    if vis:
        plt.subplot(121)
        plt.imshow(x[1].permute(1, 2, 0).detach().cpu().numpy())
        plt.subplot(122)
        plt.imshow(out[1, 0].detach().cpu().numpy())
        plt.colorbar()
        plt.show()
    return x


out_dir = "/home/lim38/dataset/imagenet-val-attack"
os.makedirs(out_dir, exist_ok=True)


i = 0  # global image count
for k, [imgs, targets] in enumerate(tqdm(loader)):
    imgs, targets = imgs.cuda(), targets.cuda()
    advs = mute_neurons(imgs, forward_pass, n_iter=40, vis=False, progress=False)
    for a in advs:
        pil = to_pil(a.detach().cpu())
        subdir, fn = dataset.imgs[i][0].split("/")[-2:]
        os.makedirs(f"{out_dir}/{subdir}", exist_ok=True)
        pil.save(f"{out_dir}/{subdir}/{fn}")
        i += 1
