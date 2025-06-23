import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import os

# Rutas
data_dir = "../dataset"
model_path = "model.pth"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset y Dataloader
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Clases
class_names = dataset.classes
print("Clases:", class_names)  # ['cat', 'dog']

# Modelo base (ResNet18 con pesos actualizados)
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 clases: cat y dog

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    acc = correct / total * 100
    print(f"Época {epoch+1}/{EPOCHS} - Accuracy: {acc:.2f}%")

# Guardar modelo
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en {model_path}")
