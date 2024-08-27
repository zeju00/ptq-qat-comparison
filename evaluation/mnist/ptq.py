import torch
import torch.nn as nn
import torch.quantization as quantization
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # tqdm을 임포트합니다.
import time  # 시간 측정을 위한 모듈
import os
from torch.quantization import QConfig

# Step 0: Define your model architecture (assuming ComplexCNN from your previous code)
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        
        # First Conv Block
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # MNIST는 흑백 이미지이므로 입력 채널을 1로 수정
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
        
        # FC Layer
        self.fc1 = nn.Linear(512, 1024)  # Flatten 이후의 크기에 맞춰 수정
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        
        # Quantization Stubs
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

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
        quantization.fuse_modules(self, [
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

# Step 1: Prepare MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=1024, shuffle=False)

# Step 2: Load the original model
model = torch.load('original_model.pt', map_location='cpu')
model.eval()

# If your model includes a fuse_model method, you should call it here
if hasattr(model, 'fuse_model'):
    model.fuse_model()

qconfig = QConfig(activation=torch.quantization.default_observer,
                  weight=torch.quantization.default_weight_observer)

# Prepare the model for quantization
model.qconfig = qconfig
quantization.prepare(model, inplace=True)

# Measure the calibration time
start_time = time.time()

# Step 3: Calibrate the model with the MNIST dataset using tqdm for progress visualization
with torch.no_grad():
    for images, _ in tqdm(test_loader, desc="Calibrating model"):
        model(images)

end_time = time.time()
calibration_time = end_time - start_time
print(f"Calibration completed in {calibration_time:.2f} seconds.")

# Convert the model to a quantized version
quantization.convert(model, inplace=True)

# Step 4: Save the quantized model
quantized_model_path = 'ptq_model.pt'
torch.save(model, quantized_model_path)

print("Quantized model saved as 'ptq_model.pt'")

# Step 5: Check the data types of model parameters to verify quantization
for name, param in model.named_parameters():
    print(f"{name} - {param.dtype}")

# Step 6: Compare the size of the original and quantized models
original_model_size = os.path.getsize('original_model.pt')
quantized_model_size = os.path.getsize(quantized_model_path)

print(f"Original model size: {original_model_size / 1024:.2f} KB")
print(f"Quantized model size: {quantized_model_size / 1024:.2f} KB")
print(f"Size reduction: {(original_model_size - quantized_model_size) / original_model_size * 100:.2f}%")

