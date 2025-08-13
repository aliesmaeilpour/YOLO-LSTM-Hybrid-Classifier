import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

class YOLOv11mLSTMClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_prob=0.5, lstm_dropout=0.3):
        super(YOLOv11mLSTMClassifier, self).__init__()
        
        self.backbone = self._create_yolov8_backbone()
        
        # Freezing strategy
        self._freeze_backbone_layers()

        self.lstm = nn.LSTM(
            input_size=512,  # YOLOv11m backbone output channels
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=lstm_dropout
        )
        
        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob/2),
            nn.Linear(128, num_classes)
        )
        
        # Weight initialization
        # self._init_weights()

    def _create_yolov8_backbone(self):
        """Create and return YOLOv8 backbone without detection heads"""

        model = YOLO('yolo11m-cls.pt')

        backbone = model.model.model[0:10]
        
        return backbone

    def _freeze_backbone_layers(self):
        """Freeze early layers, fine-tune later layers"""
        total_layers = len(list(self.backbone.children()))
        freeze_until = int(total_layers * 0.75)
        
        for i, child in enumerate(self.backbone.children()):
            if i < freeze_until:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        
        if isinstance(features, (list, tuple)):
            x = features[-1]
        else:
            x = features
        
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)
        
        x, _ = self.lstm(x)
        
        attn_weights = torch.softmax(torch.sum(x, dim=-1), dim=1)
        x = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        
        return self.classifier(x)