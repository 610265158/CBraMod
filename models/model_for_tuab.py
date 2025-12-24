import timm
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, param):
        super().__init__()

        self.model = timm.create_model('efficientnet_b0',
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

    def forward(self, x):
        bs = x.size(0)
        bz, ch_num, seq_len, patch_size = x.shape
        x = x.view(bs, 1, ch_num, -1)
        reshaped_tensor = x.view(bs, 1, ch_num, 200, 10)
        reshaped_and_permuted_tensor = reshaped_tensor.permute(0, 1, 2, 4, 3)
        x = reshaped_and_permuted_tensor.reshape(bs, 1, ch_num * 10, 200)

        x = self.model.forward_features(x)
        x = self.pool(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc1(x)

        return x
