import numpy as np
import torch 
from modules.tdtan.model_config import config
from terminaltables import AsciiTable

def get_proposal_results(scores, durations):
    # assume all valid scores are larger than one
    out_sorted_times = []
    for score, duration in zip(scores, durations):
        T = score.shape[-1]
        sorted_indexs = np.dstack(np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
        sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)

        sorted_indexs[:,1] = sorted_indexs[:,1] + 1
        sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
        target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
        out_sorted_times.append((sorted_indexs.float() / target_size * duration).tolist())

    return out_sorted_times


def iou(pred, gt): # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(gt,list)
    pred_is_list = isinstance(pred[0],list)
    gt_is_list = isinstance(gt[0],list)
    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:,0,None], gt[None,:,0])
    inter_right = np.minimum(pred[:,1,None], gt[None,:,1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:,0,None], gt[None,:,0])
    union_right = np.maximum(pred[:,1,None], gt[None,:,1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:,0]
    if not pred_is_list:
        overlap = overlap[0]
    
    return overlap

def rank(pred, gt):
    return pred.index(gt) + 1

def nms(dets, thresh=0.4, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    order = np.arange(0,len(dets),1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]

def eval(segments, data):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU,str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL,str) else [config.TEST.RECALL]

    eval_result = [[[] for _ in recalls] for _ in tious]
    max_recall = max(recalls)
    average_iou = []
    for seg, dat in zip(segments, data):
        seg = nms(seg, thresh=config.TEST.NMS_THRESH, top_k=max_recall).tolist()
        overlap = iou(seg, dat)
        overlap = np.array([overlap])
        average_iou.append(np.mean(np.sort(overlap[0])[-3:]))

        for i,t in enumerate(tious):
            for j,r in enumerate(recalls):
                eval_result[i][j].append((overlap > t)[:r].any())
    eval_result = np.array(eval_result).mean(axis=-1)
    miou = np.mean(average_iou)

    return eval_result, miou

def display_results(eval_result, miou, title=None):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU,str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL,str) else [config.TEST.RECALL]

    display_data = [['Rank@{},mIoU@{}'.format(i,j) for i in recalls for j in tious]+['mIoU']]
    eval_result = eval_result*100
    miou = miou*100
    display_data.append(['{:.02f}'.format(eval_result[j][i]) for i in range(len(recalls)) for j in range(len(tious))]
                        +['{:.02f}'.format(miou)])
    table = AsciiTable(display_data, title)
    for i in range(len(tious)*len(recalls)):
        table.justify_columns[i] = 'center'

    return table.table

def eval_predictions(segments, data, verbose=True):
    eval_result, miou = eval(segments, data)
    if verbose:
        print(display_results(eval_result, miou, ''))
    return eval_result, miou