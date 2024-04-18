# from https://github.com/Huntersxsx/MGPN

from torch import nn
import torch
import math
import numpy as np

from modules.mgpn.model_config import config
from modules.mgpn import feature_encoder
from modules.mgpn import choice_generator
from modules.mgpn import finegrained_encoder
from modules.mgpn import modality_interactor
from modules.mgpn import relation_constructor
import modules.mgpn.loss as loss

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.encoder_layer = getattr(feature_encoder, config.MGPN.COARSE_GRAINED_ENCODER.NAME)(config.MGPN.COARSE_GRAINED_ENCODER.PARAMS)
        self.interactor_layer1 = getattr(modality_interactor, config.MGPN.COATTENTION_MODULE.NAME)(config.MGPN.COATTENTION_MODULE.PARAMS)
        self.generator_layer = getattr(choice_generator, config.MGPN.CHOICE_GENERATOR.NAME)(config.MGPN.CHOICE_GENERATOR.PARAMS)
        self.finegrained_layer = getattr(finegrained_encoder, config.MGPN.FINE_GRAINED_ENCODER.NAME)(config.MGPN.FINE_GRAINED_ENCODER.PARAMS)
        self.interactor_layer2 = getattr(modality_interactor, config.MGPN.CONDITIONED_INTERACTION_MODULE.NAME)(config.MGPN.CONDITIONED_INTERACTION_MODULE.PARAMS)
        self.relation_layer = getattr(relation_constructor, config.MGPN.CHOICE_COMPARISON_MODULE.NAME)(config.MGPN.CHOICE_COMPARISON_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.MGPN.PRED_INPUT_SIZE, 1, 1, 1)

        # self.pred_layer2 = nn.Conv2d(config.RANET.PRED_INPUT_SIZE, 1, 1, 1)

    def forward(self, textual_input, textual_mask, visual_input, map_gt):
        vis_encoded, txt_encoded = self.encoder_layer(visual_input, textual_input, textual_mask)
        vis_fused, txt_fused = self.interactor_layer1(vis_encoded, txt_encoded)
        boundary_map, content_map, map_mask = self.generator_layer(vis_fused)
        vis_finegrained, txt_finegrained = self.finegrained_layer(vis_fused, txt_fused)
        # boundary_map, content_map, map_mask = self.generator_layer(vis_finegrained)
        fused_map = self.interactor_layer2(boundary_map, content_map, map_mask, vis_finegrained, txt_finegrained)
        # fused_map = torch.cat((boundary_map, content_map), dim=1) * map_mask.float()
        relation_map = self.relation_layer(fused_map, boundary_map, content_map, map_mask)
        # relation_map = self.relation_layer(fused_map, map_mask)   
        score_map = self.pred_layer(relation_map) * map_mask.float() 

        # score_map = self.pred_layer(content_map) * map_mask.float()

        # coarse_score = self.pred_layer2(boundary_map + content_map) * map_mask.float() 
        # score_map = torch.sigmoid(coarse_score) * score_map * map_mask.float() 
        # score_map = 2 * score_map + coarse_score

        map_gt = map_gt.unsqueeze(dim=1)
        loss_value, joint_prob = getattr(loss, config.LOSS1.NAME)(score_map, map_mask, map_gt, config.LOSS1.PARAMS)
        
        return joint_prob, loss_value #score_map, map_mask


