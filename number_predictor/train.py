import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10
model_save_path = 'model.pth'

class Binarize(object):
    def __call__(self, img):
        img = transforms.ToTensor()(img)  # Convert to tensor
        img = img * 255  # Scale to [0, 255]
        img = (img >= 127.5).float() * 255  # Binarize: 0 or 255
        return img

# 2. Transformations
# MNIST images are 28x28, normalize them to [0, 1] range and then standardize to zero mean and unit variance
transform = transforms.Compose([
    Binarize(),  # Custom binarization
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] (optional)
])

# 3. Load the MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# 4. Split the dataset into 70% training and 30% validation sets
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the images
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 6. Initialize the model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 7. Training loop
def train(model, train_loader):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 8. Evaluation function
def evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 9. Train the model
train(model, train_loader)

# 10. Evaluate the model
evaluate(model, test_loader)

torch.save(model.state_dict(), model_save_path)

