from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import Model
from torch.utils.data import Dataset
import os
import cv2
import tqdm

EMOTIONS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
PATH = "data/fer2013"

def load_fer_data(data_path):
    images, labels = [], []
    for label_idx, emotion in EMOTIONS.items():
        # Check for both capitalized and lowercase folder names just in case
        folder_path = os.path.join(data_path, emotion.lower())
        if not os.path.exists(folder_path):
            folder_path = os.path.join(data_path, emotion)
            
        if not os.path.exists(folder_path): 
            print(f"Warning: Folder not found for {emotion}")
            continue
            
        for img_name in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, img_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label_idx)
    return np.array(images), np.array(labels)



class FER2013Dataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.data, self.labels = load_fer_data(os.path.join(PATH, split))
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        image, label = torch.from_numpy(image).unsqueeze(0).float() / 255.0, torch.tensor(label)  # Normalize and convert to tensor

        if self.transform:
            image = self.transform(image)
        return image, label
    

train_dataset = FER2013Dataset(split="train")
test_dataset = FER2013Dataset(split="test")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = "mps"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_classes=len(EMOTIONS)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
eval_every = 5
for epoch in tqdm.tqdm(range(num_epochs), desc="Training"):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    if (epoch + 1) % eval_every == 0:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm.tqdm(test_loader, desc="Evaluating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")