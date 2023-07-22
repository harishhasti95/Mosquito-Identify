import os
import torch, torchvision
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
from tqdm.notebook import tqdm
import torch.optim as optim
import torchvision.transforms as transform_fn
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm.notebook import tqdm
from dataset import mosquitoDatasetClassification
import warnings
warnings.filterwarnings('ignore')

classes = ['albopictus', 'culex', 'anopheles', 'culiseta', 'japonicus/koreicus', 'aegypti']
reverse_encoding = {index:value for index, value in enumerate(classes)}


imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean and std values of the Imagenet Dataset so that pretrained models could also be used
transform_fn = transform_fn.Compose([transform_fn.Resize((256, 256)),
                        transform_fn.ToTensor(),
                        transform_fn.Normalize(*imagenet_stats)])


transformed_dataset = mosquitoDatasetClassification(csv_file='train.csv', root_dir='train_images/', transform = transform_fn)

# for i, sample in enumerate(transformed_dataset):
#     print(i, sample['image'].size(), sample['label'].size())
#     if i == 3:
#         break


val_percent = int(0.99 * len(transformed_dataset)) 
train_size = len(transformed_dataset) - val_percent
val_size = len(transformed_dataset) - train_size

train_ds, val_ds = random_split(transformed_dataset, [train_size, val_size])


# dataloader = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True, num_workers=0)

print(len(train_ds), len(val_ds))


batch_size = 4
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

print(len(transformed_dataset))

# train_label_counts = {i: 0 for i in range(6)}
# val_label_counts = {i: 0 for i in range(6)}

# for each_train_sample in train_loader:
#     temp_numpy = each_train_sample['label'].numpy()
#     for i in temp_numpy:
#         i = list(i)
#         train_label_counts[i.index(1)] += 1
#         # for j in range(len(i)):
#         #     if i[j] == 1:
#         #         train_label_counts[j] += 1

# for each_val_sample in val_loader:
#     temp_numpy = each_val_sample['label'].numpy()
#     for i in temp_numpy:
#         i = list(i)
#         val_label_counts[i.index(1)] += 1
#         # for j in range(len(i)):
#         #     if i[j] == 1:
#         #         train_label_counts[j] += 1
# print(train_label_counts)
# print(val_label_counts)
    

# Set device
device = torch.device("mps" if torch.cuda.is_available() else "cpu")



# Hyperparameters
in_channel = 3
num_classes = 6
learning_rate = 3e-4
num_epochs = 1


# Model
model = torchvision.models.googlenet(weights="DEFAULT")

# freeze all layers, change final linear layer with num_classes
for param in model.parameters():
    param.requires_grad = False

# final layer is not frozen
model.fc = nn.Linear(in_features=1024, out_features=num_classes)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, sampler in enumerate(train_loader):
        data = sampler['image']
        targets = sampler['label']
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        print('Epoch number', epoch, 'Batch Idx - ', batch_idx, ',', str(batch_idx / len(train_loader)) + ' percent done')
        print(str(batch_idx / len(train_loader)) + ' percent done')
        losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for batch, sampler in enumerate(loader):
            x = sampler['image']
            y = sampler['label'].argmax(1)
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(val_loader, model)