import torch.nn as nn

from src.models import modeling
from src.models import _deeplab
from src.models.registry import seg_net


@seg_net.register
class RgbSeg(nn.Module):
    def __init__(self, seg_model, num_classes, output_stride, separable_conv, pretrained_backbone):
        super(RgbSeg, self).__init__()
        self.segNet = modeling.__dict__[seg_model](num_classes=num_classes,
                                                   output_stride=output_stride,
                                                   pretrained_backbone=pretrained_backbone)
        if separable_conv and 'plus' in seg_model:
            _deeplab.convert_to_separable_conv(self.segNet.classifier)

    def forward(self, vis):
        logit_vi = self.segNet(vis)

        return logit_vi


if __name__ == '__main__':
    # Test the RgbSeg class
    import torch
    seg_model = 'deeplabv3plus_resnet101'  # Example segmentation model
    num_classes = 10  # Number of output classes
    output_stride = 16  # Output stride
    separable_conv = False  # Whether to use separable convolutions
    pretrained_backbone = False  # Whether to use a pretrained backbone

    # Create an instance of RgbSeg
    seg_net_instance = seg_net['RgbSeg'](seg_model, num_classes, output_stride, separable_conv, pretrained_backbone)
    print(seg_net_instance)
    # Generate a random input tensor with appropriate dimensions
    batch_size = 2
    input_channels = 3
    height = 512
    width = 512
    input_tensor = torch.randn(batch_size, input_channels, height, width)

    # Forward pass
    output = seg_net_instance(input_tensor)
    print(output.shape)  # Print the shape of the output tensor
