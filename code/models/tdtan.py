import torch
from torch import nn
from modules.tdtan.model_config import config
from modules.tdtan import frame_modules
from modules.tdtan import prop_modules
from modules.tdtan import map_modules
from modules.tdtan import fusion_modules
import modules.tdtan.loss as loss

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)

    def extract_features(self, textual_input, textual_mask, visual_input) :
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)

        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask

        return fused_h, prediction, map_mask

    def forward(self, textual_input, textual_mask, visual_input, map_gt) :
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)
        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask
        loss_value, joint_prob = getattr(loss, config.LOSS.NAME)(prediction, map_mask, map_gt, config.LOSS.PARAMS)

        return joint_prob, loss_value