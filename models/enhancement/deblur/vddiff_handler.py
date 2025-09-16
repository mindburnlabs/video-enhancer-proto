#!/usr/bin/env python3
"""
VDDiff Deblurring Handler
"""

import torch
import torch.nn as nn

class VDDiff(nn.Module):
    def __init__(self):
        super(VDDiff, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1)
        )

    def forward(self, x):
        return self.main(x)

class VDDiffHandler:
    def __init__(self, model_path="", device="cpu"):
        self.model = VDDiff()
        # self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()

    def deblur(self, image):
        with torch.no_grad():
            return self.model(image)