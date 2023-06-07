import torch
from torch import nn
from modules.ranet.model_config import config
from modules.ranet import feature_encoder
from modules.ranet import choice_generator
from modules.ranet import modality_interactor
from modules.ranet import relation_constructor
import modules.ranet.loss as loss

import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.encoder_layer = getattr(feature_encoder, config.RANET.ENCODER_LAYER.NAME)(config.RANET.ENCODER_LAYER.PARAMS)
        self.generator_layer = getattr(choice_generator, config.RANET.GNERATOR_LAYER.NAME)(config.RANET.GNERATOR_LAYER.PARAMS)
        self.interactor_layer = getattr(modality_interactor, config.RANET.INTERACTOR_LAYER.NAME)(config.RANET.INTERACTOR_LAYER.PARAMS)
        self.relation_layer = getattr(relation_constructor, config.RANET.RELATION_LAYER.NAME)(config.RANET.RELATION_LAYER.PARAMS)
        self.pred_layer = nn.Conv2d(config.RANET.PRED_INPUT_SIZE, 1, 1, 1)
        
    def forward(self, textual_input, textual_mask, visual_input, map_gt):

        vis_h, txt_h = self.encoder_layer(visual_input, textual_input, textual_mask) 
        choice_map, map_mask = self.generator_layer(vis_h) 
        fused_map = self.interactor_layer(choice_map, txt_h)  
        relation_map = self.relation_layer(fused_map, map_mask)  
        score_map = self.pred_layer(relation_map) * map_mask.float()  
        loss_value, joint_prob = getattr(loss, config.LOSS1.NAME)(score_map, map_mask, map_gt, config.LOSS1.PARAMS)

        return joint_prob, loss_value
    
    
