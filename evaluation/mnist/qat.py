import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub, fuse_modules, prepare_qat, convert, QConfig
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
from torchinfo import summary

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Step 1: Define your model architecture (adjusted for MNIST dataset)
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
        
        # Third Conv Block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth Conv Block
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # FC Layer (adjusted for MNIST input size)
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
        
        # Fourth Conv Block
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

# Step 3: Training (1 epoch)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=1024, shuffle=True)

# Step 2: Prepare the model for QAT
model = ComplexCNN().to(device)
model = torch.load('original_model.pt', map_location=device)
model.eval()

# 모델의 필요한 모듈을 퓨즈
if hasattr(model, 'fuse_model'):
    model.fuse_model()

# 모델을 QAT 준비
model.train()

# Custom QConfig to avoid per_channel_affine
qconfig = QConfig(activation=torch.quantization.default_observer,
                  weight=torch.quantization.default_weight_observer)

model.qconfig = qconfig
prepare_qat(model, inplace=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
for epoch in range(1):  # 1 epoch만 수행
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)  # 데이터를 GPU로 전송
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
end_time = time.time()
print(f'Training time: {end_time - start_time:.2f} seconds')

# Step 4: Convert and save the quantized model
model.eval()
convert(model, inplace=True)
torch.save(model, 'qat_model.pt')

print("QAT quantized model saved as 'qat_model.pt'")

# Step 5: Compare the size of the original and quantized models
original_model_size = os.path.getsize('original_model.pt')
quantized_model_size = os.path.getsize('qat_model.pt')

print(f"Original model size: {original_model_size / 1024:.2f} KB")
print(f"Quantized model size: {quantized_model_size / 1024:.2f} KB")
print(f"Size reduction: {(original_model_size - quantized_model_size) / original_model_size * 100:.2f}%")

print("Quantized Model Structure:")
print(model)

