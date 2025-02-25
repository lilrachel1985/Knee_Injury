import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from vit_model import VisionTransformer
from deit_model import DeiT
from swin_model import SwinTransformer
from cnn_model import CNN

# Load data
X_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')

# Convert to PyTorch tensors
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
y_val = torch.tensor(y_val, dtype=torch.long)

# Create data loader
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

def evaluate_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['vit', 'deit', 'swin', 'cnn'], help='Model name')
    args = parser.parse_args()

    if args.model == 'vit':
        model = VisionTransformer(num_classes=2)
        model.load_state_dict(torch.load('models/vit_model.pth'))
    elif args.model == 'deit':
        model = DeiT(num_classes=2)
        model.load_state_dict(torch.load('models/deit_model.pth'))
    elif args.model == 'swin':
        model = SwinTransformer(num_classes=2)
        model.load_state_dict(torch.load('models/swin_model.pth'))
    else:
        model = CNN(num_classes=2)
        model.load_state_dict(torch.load('models/cnn_model.pth'))

    accuracy, precision, recall, f1 = evaluate_model(model, val_loader)

    print(f'Model: {args.model}')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1-Score: {f1 * 100:.2f}%')