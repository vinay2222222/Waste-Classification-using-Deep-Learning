import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def get_dataset(data_dir):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return dataset, dataset.classes
#   MODEL: RWC-Net
class RWCNet(nn.Module):
    def __init__(self, num_classes=6):
        super(RWCNet, self).__init__()
        densenet = models.densenet201(pretrained=True)
        mobilenet = models.mobilenet_v2(pretrained=True)

        # Remove final classifiers
        self.densenet = nn.Sequential(*list(densenet.children())[:-1])
        self.mobilenet = nn.Sequential(*list(mobilenet.children())[:-1])

        # Fusion classifier
        self.fc = nn.Sequential(
            nn.Linear(1920 + 1280, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        # Auxiliary outputs
        self.aux1 = nn.Linear(1920, num_classes)
        self.aux2 = nn.Linear(1280, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        d_out = self.densenet(x)
        d_out = torch.flatten(d_out, 1)

        m_out = self.mobilenet(x)
        m_out = torch.flatten(m_out, 1)

        combined = torch.cat((d_out, m_out), dim=1)
        out = self.fc(combined)

        aux_out1 = self.aux1(d_out)
        aux_out2 = self.aux2(m_out)

        return self.log_softmax(out), aux_out1, aux_out2


def train_one_fold(model, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, aux1, aux2 = model(inputs)

            # Weighted losses
            loss_main = criterion(outputs, labels)
            loss_aux1 = criterion(aux1, labels) * 0.5
            loss_aux2 = criterion(aux2, labels) * 0.25
            loss = loss_main + loss_aux1 + loss_aux2

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out, _, _ = model(x)
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(y.cpu().numpy())

        acc = accuracy_score(trues, preds)
        print(f"Epoch {epoch+1}: Val Acc = {acc:.4f}")
    return model


def run_kfold(data_dir, num_folds=5, epochs=5, batch_size=32):
    dataset, classes = get_dataset(data_dir)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n===== Fold {fold+1}/{num_folds} =====")
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = RWCNet(num_classes=len(classes)).to(device)
        trained_model = train_one_fold(model, train_loader, val_loader, epochs=epochs)

    print("K-Fold training completed.")

if __name__ == "__main__":
    data_dir = "/data/data-resized"   # update path if needed
    run_kfold(data_dir, num_folds=5, epochs=5, batch_size=32)
