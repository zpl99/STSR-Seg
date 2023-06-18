import torch.nn as nn
import torch.nn.functional as F
import torch
import Nets.pac as pac
import numpy as np


class ESPC(nn.Module):
    def __init__(self, input_channel=4, upscale_factor=4):
        super(ESPC, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x, image_name=None):
        x = F.tanh(self.conv1(x))
        # np.save(f"./middle_feature/{image_name[0]}", x[0].detach().cpu().numpy())
        x = F.tanh(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x


class ESPC_tem(nn.Module):
    def __init__(self, input_channel=4, upscale_factor=4):
        super(ESPC_tem, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(input_channel, 64, (5, 5), (1, 1), (2, 2))
        # self.conv1 = pac.PacConv2d(input_channel, 64, 5, 1, 2)
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        # self.conv3 = pac.PacConv2d(32, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        # self.tem = pac.PacConv2d(3, 3, 3, padding=1, kernel_type="cosine")

    def forward(self, x, image_name=None):
        # size = x.size()
        # image = x.detach()
        x = F.tanh(self.conv1(x))
        # np.save(f"./middle_feature/{image_name[0]}", x[0].detach().cpu().numpy())
        x = F.tanh(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        # x = self.tem(x, F.upsample(image[:, :3, :, :], size[2] * self.upscale_factor)) #TODO：修改为支持任意格式输入而非方形图像
        return x


if __name__ == "__main__":
    model = ESPC_tem(input_channel=4, upscale_factor=4).cuda()
    x = torch.randn((1, 4, 64, 64)).cuda()
    out = model(x)
    print(out.shape)
    print(model)
