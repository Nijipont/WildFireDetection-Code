import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ตรวจสอบการใช้งาน GPU (ถ้ามี)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# การเตรียมข้อมูล
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = 'D:\Work\WildFireDetection\Dataset'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# สร้างโมเดล CNN
class WildfireDetectionModel(nn.Module):
    def __init__(self):
        super(WildfireDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = WildfireDetectionModel().to(device)

# กำหนดค่า loss function และ optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# เทรนโมเดล
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")

# บันทึกโมเดล
torch.save(model.state_dict(), "wildfire_detection_model.pth")
