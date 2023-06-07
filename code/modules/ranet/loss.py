import torch
import torch.nn.functional as F

def bce_rescale_loss(scores, masks, targets, cfg):
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    joint_prob = torch.sigmoid(scores) * masks.float()
    target_prob = ((targets - min_iou) / (max_iou - min_iou)).clamp(0, 1)
    target_prob = target_prob.unsqueeze(1)
    loss_value = F.binary_cross_entropy_with_logits(scores.masked_select(masks.bool()), target_prob.masked_select(masks.bool()))

    return loss_value, joint_prob
