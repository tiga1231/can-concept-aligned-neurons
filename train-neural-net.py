import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet50_Weights
from torch import nn

from tqdm.auto import tqdm
from IPython.display import clear_output
import random
import os


import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.style.use("seaborn-v0_8-colorblind")
from natsort import natsorted
from glob import glob


# Set parameters
# TODO use yaml
image_folder = "/home/lim38/dataset/imagenet-val/"
num_epochs = 100
batch_size = 64
learning_rate = 0.001  # Adam
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "resnet50"
dataset_name = "imagenet-val"
# data_split = 'all'
vis = True


# transforms
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# dataset
train_dataset = torchvision.datasets.ImageFolder(
    root=image_folder,
    transform=transform,
)
val_dataset = torchvision.datasets.ImageFolder(
    root=image_folder,
    transform=val_transform,
)


# TODO Randomly split classes into two subsets
# image_class_directories = natsorted(glob(f'{image_folder}/*/'))
# random.seed(0)
# n_classes = len(image_class_directories)
# class_set1 = set(random.sample(range(n_classes), k=500)) # {0,1,2,3,8,9,12,...} w/ seed(0)
# class_set2 = set(range(n_classes)) - set(class_set1) # complement of set1
# print('classes in set1:', class_set1)
# print('classes in set2:', class_set2)
# image_subset1 = [i for i, [img, label] in enumerate(train_dataset.imgs) if label in class_set1]
# image_subset2 = [i for i, [img, label] in enumerate(train_dataset.imgs) if label in class_set2]
# train_dataset1 = torch.utils.data.Subset(train_dataset, image_subset1)
# train_dataset2 = torch.utils.data.Subset(train_dataset, image_subset2)


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
)


# ## Model
model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# model = torch.nn.DataParallel(model) # Parallelize training across multiple GPUs
model = model.to(device)

# optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def evaluate(model, dataset, batch_size=64, device=device):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    softmax = nn.Softmax(dim=1)
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = softmax(logits).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.shape[0]
    accuracy = correct / total
    return dict(correct=correct, total=total, accuracy=accuracy)


epoch_bar = tqdm(total=num_epochs)
batch_bar = tqdm(total=len(train_loader))

# Train the model...
losses = []
loss_prev = None
for epoch in range(num_epochs):
    batch_bar.reset()
    model.train()
    for i, [inputs, labels] in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss = loss.item()
        running_average_loss = (
            loss_prev * 0.99 + loss * 0.01 if loss_prev is not None else loss
        )
        loss_prev = loss
        losses.append(running_average_loss)

        batch_bar.update()  # +1
    batch_bar.refresh()  # force finish
    torch.save(model.state_dict(), f"checkpoint_{model_name}_{dataset_name}.pth")

    # vis
    if vis:
        plt.plot(losses)
        plt.savefig("loss.png", bbox_inches="tight")
        plt.close()

    # By the end of every epoch, print the loss and accuracy,
    if type(loss) is not float:
        loss = loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, minibatch loss: {loss:.4f}")
    # TODO log loss and accuracy
    model.eval()
    evaluation = evaluate(model, val_dataset)
    correct, total, accuracy = (
        evaluation["correct"],
        evaluation["total"],
        evaluation["accuracy"],
    )
    print(f"top1 accuracy: {accuracy:.4f} ({correct}/{total})")

    epoch_bar.update()  # +1

print(f"Finished Training, Loss: {loss:.4f}")
