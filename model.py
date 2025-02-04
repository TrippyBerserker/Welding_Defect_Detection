import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix  # Import confusion matrix

train_path = "your_path"
test_path = "your_path"

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),         # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
])

# Custom dataset class for YOLO-based structure
class WeldingDefectDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        # Load the corresponding label
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")
        with open(label_path, "r") as file:
            lines = file.readlines()

        # Parse YOLO labels and map to defect/no_defect
        defect_label = 0  # Default to defect
        for line in lines:
            class_id = int(line.split()[0])
            if class_id == 1:
                defect_label = 1  # No defect
                break
            elif class_id == 0 or class_id == 2:
                defect_label = 0  # Defect
                break

        if self.transform:
            image = self.transform(image)

        return image, defect_label


# Load training and testing datasets
train_dataset = WeldingDefectDataset(
    images_dir=os.path.join(train_path, "images"),
    labels_dir=os.path.join(train_path, "labels"),
    transform=transform
)

test_dataset = WeldingDefectDataset(
    images_dir=os.path.join(test_path, "images"),
    labels_dir=os.path.join(test_path, "labels"),
    transform=transform
)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a simple ANN model
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(128 * 128 * 1, 512)  # Adjusted for grayscale (1 channel)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # Binary classification: defect/no defect
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = ANNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model():
    num_epochs = 18
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train

        # Evaluate on the test dataset
        model.eval()
        correct_test = 0
        total_test = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

                # Store predictions and labels for confusion matrix
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        test_accuracy = 100 * correct_test / total_test
        avg_loss = running_loss / len(train_loader)

        # Calculate and print the confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(f"Confusion Matrix for Epoch {epoch + 1}:\n{cm}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as 'model.pth'")

if __name__ == '__main__':
    train_model()
