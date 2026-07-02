import kagglehub
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import pickle

def main():
    print("Downloading dataset...")
    data_dir = kagglehub.dataset_download("jayaprakashpondy/soil-image-dataset")
    print(f"Dataset downloaded to {data_dir}")
    
    # The dataset might be nested in a subdirectory like 'Soil_Dataset'
    target_dir = None
    for root, dirs, files in os.walk(data_dir):
        # We look for a directory that directly contains class folders like "Black Soil"
        if any("Black" in d or "Black Soil" in d for d in dirs):
            target_dir = root
            break
            
    if not target_dir:
        target_dir = data_dir
        
    print(f"Using image directory: {target_dir}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(target_dir, transform=transform)
    print("Classes:", dataset.classes)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the classifier block (the last 2 layers are part of model.classifier)
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # Replace the final layer to output 4 classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(dataset.classes))
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Optimize only the classifier parameters since the rest is frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
        
    # Save model state and class names mapping
    model_data = {
        "state_dict": model.state_dict(),
        "classes": dataset.classes
    }
    with open("model_soil_image.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Model saved to model_soil_image.pkl")

if __name__ == "__main__":
    main()
