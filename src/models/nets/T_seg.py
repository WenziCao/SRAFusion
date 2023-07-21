import torch
import torch.nn as nn

from src.models import modeling
from src.models import _deeplab
from torch.nn import functional as F
from src.models.registry import seg_net


@seg_net.register
class TSeg(nn.Module):
    def __init__(self, seg_model, num_classes, output_stride, separable_conv, pretrained_backbone):
        super(TSeg, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(3)
        self.relu = F.relu
        self.segNet = modeling.__dict__[seg_model](num_classes=num_classes,
                                                   output_stride=output_stride,
                                                   pretrained_backbone=pretrained_backbone)
        if separable_conv and 'plus' in seg_model:
            _deeplab.convert_to_separable_conv(self.segNet.classifier)

    def forward(self, ir):
        ir = self.conv(ir)
        ir = self.bn(ir)
        ir = self.relu(ir)
        logit_vi = self.segNet(ir)

        return logit_vi


if __name__ == '__main__':
    inr = torch.rand(4, 1, 480, 640)



