# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub, fuse_modules, prepare_qat, convert
import torchvision
import torchvision.transforms as transforms
import os
import time
from tqdm import tqdm

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()

        # First Conv Block
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Conv Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Thrid Conv Block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4th Conv Block
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # FC Layer
        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        # Quantization Stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # Start quantization
        x = self.quant(x)

        # First Conv Block
        x = self.pool1(self.relu2(self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x)))))))

        # Second Conv Block
        x = self.pool2(self.relu4(self.bn4(self.conv4(self.relu3(self.bn3(self.conv3(x)))))))

        # Third Conv Block
        x = self.pool3(self.relu6(self.bn6(self.conv6(self.relu5(self.bn5(self.conv5(x)))))))

        # 4th Conv Block
        x = self.pool4(self.relu8(self.bn8(self.conv8(self.relu7(self.bn7(self.conv7(x)))))))

        # Flatten
        x = x.reshape(x.size(0), -1)

        # FC Layer
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)

        # Dequantization
        x = self.dequant(x)
        return x

    def fuse_model(self):
        # Merge BatchNorm and ReLU
        fuse_modules(self, [
            ['conv1', 'bn1', 'relu1'],
            ['conv2', 'bn2', 'relu2'],
            ['conv3', 'bn3', 'relu3'],
            ['conv4', 'bn4', 'relu4'],
            ['conv5', 'bn5', 'relu5'],
            ['conv6', 'bn6', 'relu6'],
            ['conv7', 'bn7', 'relu7'],
            ['conv8', 'bn8', 'relu8']
        ], inplace=True)

# Instantiate the model
model = ComplexCNN()

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)

# Model, Loss, Optimizer
model = ComplexCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Pre-QAT Training
model.train()

start_time = time.time()
for epoch in range(2):
    for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/2", unit="batch"):
        data, target = data.to(device), target.to(device)  # 데이터를 GPU로 이동
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
end_time = time.time()
print(f"Original Model Training Time: {(end_time - start_time) / 2:.2f} seconds")

torch.save(model, "original_model.pt")

# Check parameters
for name, param in model.named_parameters():
    print(f"Parameter name: {name}")
    print(param.data)
    print(param.size())
    print("----------")


# Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for inference
        for data, target in test_loader:
            data, target = data.to('cpu'), target.to('cpu')
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Dataset and DataLoader for Evaluation
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Load the original model
model = torch.load("original_model.pt")
model.to('cpu')

# Evaluate the model
accuracy = evaluate_model(model, test_loader)
print(f"Model Accuracy: {accuracy:.2f}%")


