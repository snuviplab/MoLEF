# MIT License

# Copyright (c) 2022 Sangmin Woo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# from https://github.com/sangminwoo/Explore-And-Match

import torch
import torch.nn as nn
from modules.lvtr.modeling.backbone import build_backbone
from modules.lvtr.modeling.lvtr import build_lvtr
from modules.lvtr.modeling.loss import build_loss

class Model(nn.Module):

	def __init__(self, args):
		super(Model, self).__init__()
		self.backbone = build_backbone(args)
		self.head = build_lvtr(args)
		self.skip_backbone = args.data_type=='features'
		self.criterion = build_loss(args)

	def forward(self, src_txt, src_vid, src_txt_mask=None, src_vid_mask=None, targets=None,
				att_visualize=False, corr_visualize=False, epoch_i=None, idx=None):
		if not self.skip_backbone:
			src_txt, src_txt_mask, src_vid, src_vid_mask = \
				self.backbone(src_txt, src_vid)  # BxNx512, BxMx512

		# add no_class token
		# B, N, D = src_txt.shape
		# no_class_token = torch.zeros((B, 1, D), device=src_txt.device)  # Bx1x512
		# no_class_mask = torch.zeros((B, 1), dtype=torch.float32, device=src_txt_mask.device)  # Bx1
		# src_txt = torch.cat([src_txt, no_class_token], dim=1)  # Bx(N+1)x512
		# src_txt_mask = torch.cat([src_txt_mask, no_class_mask], dim=1)  # Bx(N+1)

		outputs = self.head(
			src_txt, src_txt_mask,
			src_vid, src_vid_mask,
			att_visualize, corr_visualize, epoch_i, idx
		)

		# print(outputs.shape)
		# print(targets.shape)
		loss_dict = self.criterion(outputs, targets)
		weight_dict = self.criterion.weight_dict
		loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

		return outputs, loss