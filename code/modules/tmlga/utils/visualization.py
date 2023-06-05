import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from modules.tmlga.utils.miscellaneous import mkdir
from modules.tmlga.config import cfg 

class Visualization(object):
    def __init__(self, dataset_size, is_train=True):

        self.loss = []
        self.IoU  = []
        self.mIoU = []
        self.aux_mIoU = []
        self.individual_loss = {}
        self.vis_dir = "{}{}".format(cfg.VISUALIZATION_DIRECTORY, cfg.EXPERIMENT_NAME)
        print(self.vis_dir)
        mkdir(self.vis_dir)
        self.cfg = cfg
        self.s_samples = np.random.randint(dataset_size, size=4)
        self.s_samples = np.insert(self.s_samples,0, 100)

        for s in self.s_samples:
            self.individual_loss[str(s)] = []
            mkdir("{}/{}".format(self.vis_dir, str(s)))
        if is_train == True:
            self.state = "training"
        else:
            self.state = "testing"

    def tIoU(self, start, end, pred_start, pred_end):
        tt1 = np.maximum(start, pred_start)
        tt2 = np.minimum(end, pred_end)
        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clip(0)
        # Segment union.
        segments_union = (pred_end - pred_start) \
          + (end - start) - segments_intersection
        # Compute overlap as the ratio of the intersection
        # over union of two segments.
        tIoU = segments_intersection.astype(float) / segments_union
        return tIoU

    def run(self, indexes, pred_start, pred_end, start, end, lengths,
                epoch, loss, individual_loss, attention, atten_loss,
                time_starts, time_ends, factors, fps):
        l = loss.detach().item()
        self.loss.append(l)

        startings = np.argmax(pred_start.detach().cpu().numpy(), axis=1)
        endings   = np.argmax(pred_end.detach().cpu().numpy(), axis=1)
        
        startings = factors * (startings) / fps
        endings = factors * (endings + 1) / fps

        gt_start = np.array(time_starts)
        gt_end   = np.array(time_ends)
        
        iou = self.tIoU(gt_start, gt_end, startings, endings)
        self.IoU.append(iou)
        mIoU = np.mean(iou)
        self.mIoU.append(mIoU)
        ret = {}
        for j, indx in enumerate(indexes):
            ret[int(indx)] = {"iou": round(iou[j], 2),
                        "p_start": round(startings[j], 2),
                        "p_end": round(endings[j], 2)}

        return ret

    def plot(self, epoch):
        new_ious = []
        for batch in self.IoU:
            for p in batch:
                new_ious.append(p)
        th = {0.1: 0, 0.3: 0, 0.5: 0, 0.7: 0}
        for i in range(len(new_ious)):
            for k in th.keys():
                if round(new_ious[i],2) >= k:
                    th[k] += 1
        if self.state == "training":
            a = {str(k): round(v * 100 / self.cfg.DATASETS.TRAIN_SAMPLES,2) for k, v in th.items()}
        else:
            a = {str(k): round(v * 100 / self.cfg.DATASETS.TEST_SAMPLES,2) for k, v in th.items()}

        self.IoU = []
        self.mIoU = []
        return a
