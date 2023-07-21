import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image, make_grid

from src.models.registry import fusion_net
from src.utils.tool import create_file


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ConvOut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvOut, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1,
                                    groups=in_channel)
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return F.leaky_relu(out, negative_slope=0.2)


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.conv1 = DepthWiseConv(in_channel=channels, out_channel=channels)
        self.conv2 = DepthWiseConv(in_channel=2 * channels, out_channel=channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        return x


class GraDenseblock(nn.Module):
    def __init__(self, channels, out_ch):
        super(GraDenseblock, self).__init__()
        self.denseblock = DenseBlock(channels=channels)
        self.conv3 = nn.Conv2d(in_channels=channels * 3, out_channels=out_ch,
                               kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv4 = Sobelxy(channels=channels)

        self.conv5 = nn.Conv2d(in_channels=channels, out_channels=out_ch,
                               kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        x_1 = self.denseblock(x)
        x_1 = self.conv3(x_1)
        x_4 = self.conv4(x)
        x_4 = self.conv5(x_4)
        return F.leaky_relu(x_1 + x_4, negative_slope=0.1)


class Encoder(nn.Module):
    def __init__(self, in_ch, channels):
        super(Encoder, self).__init__()
        self.conv0 = Conv(in_ch=in_ch, out_ch=channels)
        self.gra_dense_1 = GraDenseblock(channels=channels, out_ch=2 * channels)
        self.gra_dense_2 = GraDenseblock(channels=2 * channels, out_ch=3 * channels)

    def forward(self, x):
        x_0 = self.conv0(x)
        x_mid = self.gra_dense_1(x_0)
        x_mid_1 = self.gra_dense_2(x_mid)

        return x_mid_1


@fusion_net.register
class FusionNet(nn.Module):
    def __init__(self, output, save_fea_list):
        super(FusionNet, self).__init__()
        self.v_enc = Encoder(in_ch=1, channels=24)
        self.i_enc = Encoder(in_ch=1, channels=24)
        self.dec_4 = Conv(in_ch=144, out_ch=72)
        self.dec_3 = Conv(in_ch=72, out_ch=36)
        self.dec_2 = Conv(in_ch=36, out_ch=18)
        self.dec_1 = ConvOut(in_ch=18, out_ch=output)

        self.save_fea_list = save_fea_list

    def forward(self, image_vis, image_ir, img_name, epoch):
        # Extract visible and infrared image inputs
        x_vis_origin = image_vis[:, :1]
        x_inf_origin = image_ir

        # Encode visible and infrared images
        v_1 = self.v_enc(x_vis_origin)
        i_1 = self.i_enc(x_inf_origin)

        # Perform fusion and decode the features
        f_4 = self.dec_4(torch.cat((v_1, i_1), dim=1))
        f_3 = self.dec_3(f_4)
        f_2 = self.dec_2(f_3)
        f_1 = self.dec_1(f_2)

        if (epoch % 5 == 0 or epoch == 1) and self.save_fea_list:
            create_file(r'./visualization/save_fea/')
            create_file(r'./visualization/save_fea/' + 'epoch_{}'.format(epoch))
            # self._save_feature(v_1, img_name, 'v_1', epoch, self.save_fea_list)
            # self._save_feature(i_1, img_name, 'i_1', epoch, self.save_fea_list)
            # self._save_feature(f_4, img_name, 'f_4', epoch, self.save_fea_list)
            # self._save_feature(f_3, img_name, 'f_3', epoch, self.save_fea_list)
            # self._save_feature(f_2, img_name, 'f_2', epoch, self.save_fea_list)
            self._save_feature(f_1, img_name, 'f_1', epoch, self.save_fea_list)

        # print("All the grid maps saved successfully.")
        return f_1

    def _save_feature(self, feature, img_name, fea_name, epoch, save_fea_list):
        # Get the dimensions of the feature maps
        batch_size, channels, height, width = feature.size()

        # Normalize each channel individually
        # normalized_feature_maps = feature.new_empty(feature.size())
        # for i in range(channels):
        #     channel_min = feature[:, i].min()
        #     channel_max = feature[:, i].max()
        #     normalized_channel = (feature[:, i] - channel_min) / (channel_max - channel_min)
        #     normalized_feature_maps[:, i] = normalized_channel

        # Check if the feature should be saved for this epoch
        if fea_name in save_fea_list:
            # Create directories for saving feature maps
            save_path = r'./visualization/save_fea/' + 'epoch_{}'.format(epoch) + '/' + fea_name
            create_file(save_path)
            for i in range(batch_size):
                # Get the i-th feature map
                # feature_map = normalized_feature_maps[i]
                feature_map = feature[i]
                feature_map = feature_map.unsqueeze(1)
                save_file = os.path.join(save_path, img_name[i])
                grid = make_grid(feature_map, nrow=int(channels ** 0.5))
                save_image(grid, save_file)


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


if __name__ == '__main__':
    # Import the FusionNet class and other necessary modules

    # Define a test function
    def test_fusion_net():
        # Create an instance of the FusionNet model
        output_channels = 1
        save_fea_list = ['v_1', 'f_4']
        fusion_net = FusionNet(output_channels, save_fea_list)

        # Create some dummy input data
        batch_size = 4
        num_channels = 1
        image_height = 32
        image_width = 32
        image_vis = torch.randn(batch_size, num_channels, image_height, image_width)
        image_ir = torch.randn(batch_size, num_channels, image_height, image_width)
        img_name = ['img1', 'img2', 'img3', 'img4']
        epoch = 10

        # Call the forward method of FusionNet
        output = fusion_net.forward(image_vis, image_ir, img_name, epoch)

        # Assert the output has the correct shape
        assert output.shape == (batch_size, output_channels, image_height, image_width)

        # Additional assertions or checks based on your requirements

    # Run the test
    test_fusion_net()
