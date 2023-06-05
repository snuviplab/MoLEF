import torch
import torch.nn.functional as F

def bce_rescale_loss(scores, masks, targets, cfg):
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    joint_prob = torch.sigmoid(scores) * masks
    target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0 # target_prob (64, 64, 64) joint_prob (64, 1, 64, 64)
    # loss has nan, joint_prob not has nan, check binary cross_entropy
    loss = F.binary_cross_entropy(joint_prob.float(), target_prob.unsqueeze(1).float(), reduction='none') * masks # loss has nan

    loss_value = torch.sum(loss) / torch.sum(masks)
    return loss_value, joint_prob