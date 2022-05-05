#
__author__ = "MSc. Otso Brummer, <https://github.com/vahvero>"
__date__ = "2022-05-5"

# %% Define inputs
# pylint: disable=all

import seaborn as sb
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision
from matplotlib import pyplot as plt
from utils import load_image_to_tensor
from torch.multiprocessing import cpu_count
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

input_size = 256, 256
VERBOSE = True

"""TODO: Run with both datasets to generate
two models, change following
models spesificatation to alter models"""

# root = "binary_lymphocytes"
root = "tissue_classification"

if root == "tissue_classification":
    # Stabilise to known class order
    def find_classes(self, folder):
        return (
            [
                "empty",
                "blood",
                "cancer",
                "normal",
                "stroma",
                "other",
            ],
            {
                "empty": 0,
                "blood": 1,
                "cancer": 2,
                "normal": 3,
                "stroma": 4,
                "other": 5,
            },
        )

    ImageFolder.find_classes = find_classes

dataset = ImageFolder(
    root,
    loader=load_image_to_tensor,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomVerticalFlip(0.5),
        ],
    ),
)

# %% Create loaders

# Split dataset
n_len = len(dataset)
n_train = int(n_len * 0.8)
n_len = n_len - n_train
n_val = int(n_len * 0.5)
n_test = n_len - n_val

assert len(dataset) == n_train + n_val + n_test

print(len(dataset))
print(n_train, n_val, n_test)

train, val, test = torch.utils.data.random_split(
    dataset,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42),
)

trainloader = DataLoader(
    dataset=train,
    shuffle=True,
    batch_size=4,
    num_workers=cpu_count(),
)
valloader = DataLoader(
    dataset=val,
    shuffle=False,
    batch_size=4,
    num_workers=cpu_count(),
)
testloader = DataLoader(
    dataset=test,
    shuffle=False,
    batch_size=4,
    num_workers=cpu_count(),
)

# %% Load models
num_classes = len(dataset.classes)
learning_rate = 1e-4
device = torch.device("cuda:0")
max_epochs = 300
patience = 5
model_name = "resnet18"

models = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152,
}

model = models[model_name](pretrained=True)
best_state_dict = f"{model_name}_{root}.pth"

features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(features, features),
    nn.ReLU(),
    nn.Linear(features, num_classes),
)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = opt.Adam(
    model.parameters(),
    lr=learning_rate,
)

scheluler = StepLR(optimizer, step_size=5, gamma=0.5)

val_loss_history = [float("inf")]
val_accuracy_history = [0]

train_loss_history = [float("inf")]
train_accuracy_history = [0]

# %% Train model

for epoch in range(1, max_epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    # Train loop
    for batch, target in tqdm(trainloader, desc=f"{epoch}. Train", disable=not VERBOSE):

        batch = batch.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(batch)
        loss = loss_fn(output, target)

        loss.backward()

        optimizer.step()

        _, predicted = torch.max(output, 1)
        running_correct += torch.sum(predicted == target)

        running_loss += loss * batch.shape[0]

    epoch_loss = running_loss.item() / len(trainloader.dataset)
    epoch_acc = running_correct.item() / len(trainloader.dataset)
    train_loss_history.append(epoch_loss)
    train_accuracy_history.append(epoch_acc)

    print(
        f"{epoch}: Train loss {epoch_loss: .2f} acc {epoch_acc:.2f} ({running_correct}/{len(trainloader.dataset)})"
    )

    # Val loop
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        running_correct = 0
        for batch, target in tqdm(valloader, desc=f"{epoch}. Val", disable=not VERBOSE):

            batch = batch.to(device)
            target = target.to(device)

            output = model(batch)
            loss = loss_fn(output, target)

            running_loss += loss * batch.shape[0]

            _, predicted = torch.max(output, 1)

            running_correct += torch.sum(predicted == target)

        epoch_loss = running_loss.item() / len(valloader.dataset)
        epoch_acc = running_correct.item() / len(valloader.dataset)

        if epoch_loss < min(val_loss_history):
            print("New best model found")
            torch.save(model.state_dict(), best_state_dict)

        val_loss_history.append(epoch_loss)
        val_accuracy_history.append(epoch_acc)
        print(f"{epoch}: Val loss {epoch_loss: .2f} acc {epoch_acc:.2f}")

        if epoch > patience and val_loss_history[-patience - 1] < min(
            val_loss_history[-patience:]
        ):
            print(f"No improvement in {patience} epochs, stopping training")
            break

    scheluler.step()

train_accuracy_history = train_accuracy_history[1:]
train_loss_history = train_loss_history[1:]
val_accuracy_history = val_accuracy_history[1:]
val_loss_history = val_loss_history[1:]

# %% Create acc and loss grapsh

plt.figure()
plt.plot(train_loss_history, label="loss")
plt.plot(train_accuracy_history, label="acc")
plt.title("Train")
plt.legend()
plt.grid()
plt.xlim((0, len(train_accuracy_history) - 1))
plt.savefig(
    root + f"_{model_name}_train.png"
)

plt.figure()
plt.plot(val_loss_history, label="loss")
plt.plot(val_accuracy_history, label="acc")
plt.title("Validation")
plt.legend()
plt.grid()
plt.xlim((0, len(val_accuracy_history) - 1))
plt.savefig(
    root + f"_{model_name}_val.png"
)


# %% Run test

with torch.no_grad():
    # Test loop
    model.eval()
    # Load best model
    model.load_state_dict(torch.load(best_state_dict))
    matrix = torch.zeros([num_classes, num_classes])
    for idx, (batch, target) in tqdm(
        enumerate(testloader), desc=f"test", disable=not VERBOSE
    ):
        batch = batch.to(device)
        target = target.to(device)

        output = model(batch)

        _, predicted = torch.max(output, 1)
        for i, j in zip(predicted, target):
            matrix[i, j] += 1
    print(dataset.class_to_idx)
    print(f"correct {matrix.diag().sum() / matrix.sum() * 1e2:.2f} %")
    for row in matrix:
        print("\t".join(str(x.to(int).item()) for x in row))
    # print(matrix)
    torch.save(
        matrix, root + f"_{model_name}_test_mtx.pth"
    )

    # TODO: save as confusion matrix
    # TODO: Qualities

    confusion_matrix = (100 * matrix.T / matrix.sum(1)).T

    ticklabels = list(dataset.class_to_idx.keys())

    fig, axes = plt.subplots()
    sb.heatmap(
        confusion_matrix,
        ax=axes,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        cmap="rocket_r",
        annot=True,
        fmt="2.1f",
    )
    fig.savefig(root + f"_{model_name}_test_mtx.png")
    plt.close()

# %% Run over total dataset

dataset = ImageFolder(
    root,
    loader=load_image_to_tensor,
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.Resize(input_size)]
    ),
)
print(len(dataset), "samples")
loader = DataLoader(
    dataset,
    shuffle=False,
    batch_size=8,
)

with torch.no_grad():
    # Test loop
    model.eval()
    # Load best model
    model.load_state_dict(torch.load(best_state_dict))
    matrix = torch.zeros([num_classes, num_classes], dtype=int)
    for idx, (batch, target) in tqdm(enumerate(loader), desc=f"total", disable=True):

        batch_indices = loader.batch_size * idx
        data = dataset.imgs[batch_indices : batch_indices + loader.batch_size]

        batch = batch.to(device)
        target = target.to(device)

        output = model(batch)

        _, predicted = torch.max(output, 1)
        for i, j, d in zip(predicted, target, data):
            # if i != j:
            #     print(f"{d[0]} {j} predicted as {i}")
            matrix[i, j] += 1

    print(dataset.class_to_idx)
    print(f"correct {matrix.diag().sum() / matrix.sum() * 1e2:.2f} %\n")

    for row in matrix:
        print("\t".join(str(x.to(int).item()) for x in row))

    torch.save(
        matrix, root + f"_{model_name}_total_mtx.pth"
    )

    confusion_matrix = (100 * matrix.T / matrix.sum(1)).T

    ticklabels = list(dataset.class_to_idx.keys())

    fig, axes = plt.subplots()
    sb.heatmap(
        confusion_matrix,
        ax=axes,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        cmap="rocket_r",
        annot=True,
        fmt="2.1f",
    )
    fig.savefig(root + f"_{model_name}_total_mtx.png")
    plt.close()

# %%
