
import torch.nn as nn

class inference_framework(nn.Module):
    def __init__(self,sr, ss):
        super(inference_framework, self).__init__()
        self.sr = sr()
        self.ss = ss()

    def forward(self, input):
        lr_image = input["image"]
        hr_feature = self.sr(lr_image)
        hr_pre = self.ss(hr_feature)
        return hr_pre
