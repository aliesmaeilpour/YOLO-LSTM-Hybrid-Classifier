import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from tqdm import tqdm
from model import YOLOv11mLSTMClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLOv11mLSTMClassifier(num_classes=2).to(device)
parameters = sum(p.numel() for p  in model.parameters())
print(parameters)

checkpoint = torch.load('classification_result/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

test_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(root='classification/test', transform=test_transform)
dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

all_preds = []
all_labels = []
total_loss = 0.0
loss_fn = nn.CrossEntropyLoss()

model.eval()


with torch.no_grad():
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        total_loss = loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())




# print(wrong_classes)

total_loss = total_loss / len(dataloader.dataset)

accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

print(confusion_matrix(all_labels, all_preds))