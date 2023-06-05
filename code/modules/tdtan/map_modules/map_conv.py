import torch
from torch import nn
import torch.nn.functional as F

def get_padded_mask_and_weight(*args):
    if len(args) == 2:
        mask, conv = args
        masked_weight = torch.round(F.conv2d(mask.clone().float(), torch.ones(1, 1, *conv.kernel_size).cuda(),
                                             stride=conv.stride, padding=conv.padding, dilation=conv.dilation))
    elif len(args) == 5:
        mask, k, s, p, d = args
        masked_weight = torch.round(F.conv2d(mask.clone().float(), torch.ones(1, 1, k, k).cuda(), stride=s, padding=p, dilation=d))
    else:
        raise NotImplementedError

    masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0] #conv.kernel_size[0] * conv.kernel_size[1]
    padded_mask = masked_weight > 0

    return padded_mask, masked_weight

class MapConv(nn.Module):

    def __init__(self, cfg):
        super(MapConv, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_sizes = cfg.HIDDEN_SIZES
        kernel_sizes = cfg.KERNEL_SIZES
        strides = cfg.STRIDES
        paddings = cfg.PADDINGS
        dilations = cfg.DILATIONS
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size]+hidden_sizes
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], k, s, p, d))

    def forward(self, x, mask):
        padded_mask = mask
        for i, pred in enumerate(self.convs):
            x = F.relu(pred(x))
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
            x = x * masked_weight
        return x