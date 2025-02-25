import torch
import torch.nn as nn
import timm

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=2):
        super(SwinTransformer, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)