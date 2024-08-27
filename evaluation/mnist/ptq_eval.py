import torch
import torch.nn as nn
import torch.quantization
from torchvision import datasets, transforms

# Define the ComplexCNN class without explicitly providing scale and zero_point
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.quant = torch.quantization.QuantStub()
        
        self.conv1 = torch.nn.quantized.ConvReLU2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.Identity()

        self.conv2 = torch.nn.quantized.ConvReLU2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = torch.nn.quantized.ConvReLU2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.Identity()

        self.conv4 = torch.nn.quantized.ConvReLU2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = torch.nn.quantized.ConvReLU2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn5 = nn.Identity()

        self.conv6 = torch.nn.quantized.ConvReLU2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn6 = nn.Identity()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv7 = torch.nn.quantized.ConvReLU2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn7 = nn.Identity()

        self.conv8 = torch.nn.quantized.ConvReLU2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn8 = nn.Identity()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.quantized.Linear(512, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = torch.nn.quantized.Linear(1024, 512)
        self.fc3 = torch.nn.quantized.Linear(512, 10)

        self.dequant = torch.quantization.DeQuantStub()

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

# Load the quantized model
model = torch.load('ptq_model.pt')

# Make sure the model is fully loaded before setting to evaluation mode
model.eval()

# Define the MNIST test dataset and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Perform inference on the test dataset
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        # Perform inference
        output = model(data)
        
        # Get the predicted class
        _, predicted = torch.max(output.data, 1)
        
        # Update the total number of correct predictions
        total += target.size(0)
        correct += (predicted == target).sum().item()

# Calculate the accuracy
accuracy = 100 * correct / total
print(f'Accuracy on the MNIST test dataset: {accuracy:.2f}%')
