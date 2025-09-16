#!/usr/bin/env python3
"""
BSSTNet Deblurring Handler
"""

import torch
import torch.nn as nn

class BSSTNet(nn.Module):
    def __init__(self):
        super(BSSTNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, x):
        return self.main(x)

class BSSTNetHandler:
    def __init__(self, model_path="", device="cpu"):
        self.model = BSSTNet()
        # self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()

    def deblur(self, image):
        with torch.no_grad():
            return self.model(image)
