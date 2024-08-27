import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub, fuse_modules, prepare_qat, convert, quantize_dynamic
import torchvision
import torchvision.transforms as transforms
import time
from tqdm import tqdm
import pandas as pd
import contextlib
import random
from collections import defaultdict

# Device setup
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()

        # First Conv Block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
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
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
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
model = ComplexCNN().to(device)
model.eval()

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Function to measure inference time and accuracy
def measure_performance(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to('cpu'), target.to('cpu')
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    end_time = time.time()
    inference_time = end_time - start_time
    accuracy = 100 * correct / total
    
    print(f'Inference time: {inference_time}, Accuracy: {accuracy}')
    return inference_time, accuracy

# Perform Quantization
print("Starting Quantization...")

# Load the original pre-trained model
original_model = torch.load("original_model.pt", map_location=device)

# PTQ: Post-Training Quantization
ptq_model = torch.load("original_model.pt", map_location='cpu')

# Calibration (Static Quantization)
ptq_model.eval()
ptq_model.fuse_model()  # Fuse layers
ptq_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(ptq_model, inplace=True)

def create_balanced_subset(dataset, num_samples_per_class):
    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    selected_indices = []
    for cls, indices in class_indices.items():
        selected_indices.extend(random.sample(indices, num_samples_per_class))

    subset = torch.utils.data.Subset(dataset, selected_indices)
    return subset

start_calibration = time.time()
calibration_subset = create_balanced_subset(train_dataset, num_samples_per_class=500)
calibration_loader = torch.utils.data.DataLoader(calibration_subset, batch_size=1024, shuffle=False)

# Use the test dataset for calibration
with torch.no_grad():
    for data, _ in tqdm(calibration_loader, desc='PTQ Calibration', unit='batch'):
        ptq_model(data)
end_calibration = time.time()
print(f'Calibration Time: {end_calibration - start_calibration:.2f} seconds')

torch.quantization.convert(ptq_model, inplace=True)

# QAT: Quantization-Aware Training
qat_model = torch.load("original_model.pt", map_location=device)
qat_model.eval()
qat_model.fuse_model()  # Fuse the layers before training
qat_model.train()  # Switch back to train mode for QAT

qat_qconfig = torch.quantization.QConfig(
    activation=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine
    ),
    weight=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine
    )
)

qat_model.qconfig = qat_qconfig
torch.quantization.prepare_qat(qat_model, inplace=True)

optimizer = optim.Adam(qat_model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# 2 epoch fine-tuning for QAT
for i in range(3):
    print(f'Epoch {i+1}:')
    for data, target in tqdm(train_loader, desc="QAT Fine-tuning", unit="batch"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = qat_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

qat_model.eval()
qat_model.to('cpu')  # Move to CPU before quantization
convert(qat_model, inplace=True)

print("Quantization completed. Starting evaluation iterations...")

# Reload the original pre-trained model for each iteration
original_model = torch.load("original_model.pt", map_location='cpu')
original_model.eval()

'''
# Architecture of Models
with open('ptq_output.txt', 'w') as f:
    with contextlib.redirect_stdout(f):
        print("PTQ Model")
        for name, param in ptq_model.state_dict().items():
            print(f"Parameter name: {name}")
            print(param)
            print("----------")

with open('qat_output.txt', 'w') as f:
    with contextlib.redirect_stdout(f):
        print("QAT Model")
        for name, param in qat_model.state_dict().items():
            print(f"Parameter name: {name}")
            print(param)
            print("----------")

with open('output.txt', 'w') as f:
    with contextlib.redirect_stdout(f):
        print("Original Model")
        for name, param in original_model.named_parameters():
            print(f"Parameter name: {name}")
            print(param.data)
            print("----------")
'''

# Evaluation Iterations
results = []
for i in range(5):
    print(f"Iteration {i+1}/5")
    
    print(f'Iteration {i+1} ptq model')
    ptq_time, ptq_accuracy = measure_performance(ptq_model, test_loader)
    print(f'Iteration {i+1} qat model')
    qat_time, qat_accuracy = measure_performance(qat_model, test_loader)
    print(f'Iteration {i+1} original model')
    original_time, original_accuracy = measure_performance(original_model, test_loader)

    results.append({
        "Iteration": i + 1,
        "Original_Inference_Time": original_time,
        "Original_Accuracy": original_accuracy,
        "PTQ_Inference_Time": ptq_time,
        "PTQ_Accuracy": ptq_accuracy,
        "QAT_Inference_Time": qat_time,
        "QAT_Accuracy": qat_accuracy,
    })

# Convert results to DataFrame and compute statistics
df = pd.DataFrame(results)
df['Original_Inference_Time_Mean'] = df['Original_Inference_Time'].mean()
df['Original_Accuracy_Mean'] = df['Original_Accuracy'].mean()
df['PTQ_Inference_Time_Mean'] = df['PTQ_Inference_Time'].mean()
df['PTQ_Accuracy_Mean'] = df['PTQ_Accuracy'].mean()
df['QAT_Inference_Time_Mean'] = df['QAT_Inference_Time'].mean()
df['QAT_Accuracy_Mean'] = df['QAT_Accuracy'].mean()

df['Original_Inference_Time_Max'] = df['Original_Inference_Time'].max()
df['Original_Accuracy_Max'] = df['Original_Accuracy'].max()
df['PTQ_Inference_Time_Max'] = df['PTQ_Inference_Time'].max()
df['PTQ_Accuracy_Max'] = df['PTQ_Accuracy'].max()
df['QAT_Inference_Time_Max'] = df['QAT_Inference_Time'].max()
df['QAT_Accuracy_Max'] = df['QAT_Accuracy'].max()

df['Original_Inference_Time_Min'] = df['Original_Inference_Time'].min()
df['Original_Accuracy_Min'] = df['Original_Accuracy'].min()
df['PTQ_Inference_Time_Min'] = df['PTQ_Inference_Time'].min()
df['PTQ_Accuracy_Min'] = df['PTQ_Accuracy'].min()
df['QAT_Inference_Time_Min'] = df['QAT_Inference_Time'].min()
#df['QAT_Accuracy_Min'] = df['QAT_Accuracy_Min'].min()

# Save to CSV
df.to_csv("quantization_performance.csv", index=False)
print("Results saved to quantization_performance.csv")

