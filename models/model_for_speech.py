import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, param):
        super().__init__()

        self.model = timm.create_model('convnextv2_nano',
                                       pretrained=True,
                                       in_chans=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1280, out_features=param.num_of_classes, bias=True)

        self._init_fc_weights()

    def _init_fc_weights(self):
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)

    def _pad_to_multiple_of_32(self, x):
        _, _, h, w = x.shape

        # 计算需要填充的大小
        pad_h = (32 - h % 32) % 32  # 如果h已经是32的倍数，则不需要填充
        pad_w = (32 - w % 32) % 32  # 如果w已经是32的倍数，则不需要填充

        # 计算上下左右的填充大小
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # 使用F.pad进行填充 (左, 右, 上, 下)
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        return x
    def forward(self, x):
        bs = x.size(0)
        bz, ch_num, seq_len, patch_size = x.shape
        x = x.view(bs, 1, ch_num, -1)
        # reshaped_tensor = x.view(bs, 1, ch_num, 200, 4)
        # reshaped_and_permuted_tensor = reshaped_tensor.permute(0, 1, 2, 4, 3)
        # x = reshaped_and_permuted_tensor.reshape(bs, 1, ch_num * 4, 200)
        x = self._pad_to_multiple_of_32(x)
        x = self.model.forward_features(x)
        x = self.pool(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc1(x)

        return x
