import torch
import numpy as np

class BatchCollator(object):

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))

        index = transposed_batch[0]
        vid_feats = transposed_batch[1]
        video_mask = transposed_batch[2]
        words_vec = transposed_batch[3]
        word_mask = transposed_batch[4]
        fr_label = transposed_batch[5]
        scores = transposed_batch[6]
        scores_mask = transposed_batch[7]
        id2pos = transposed_batch[8]
        node_mask = transposed_batch[9]
        adj_mat = transposed_batch[10]
        map_gt = transposed_batch[11]
        duration = transposed_batch[12]
        gt = transposed_batch[13]
        sample_index = transposed_batch[14]
        sub_feats = transposed_batch[22]
        sub_mask = transposed_batch[23]
        tef_feats = transposed_batch[24]

        # origin batch
        # below items with variable length for tmlga model 

        # raw_vid_feats = transposed_batch[15]
        # raw_words_vec = transposed_batch[16]
        # loc_start = transposed_batch[17]
        # loc_end = transposed_batch[18]
        # localization = transposed_batch[19]
        # raw_vid_feats_length = transposed_batch[20]
        # factors = transposed_batch[21]


        return index, torch.tensor(np.array(vid_feats)), torch.tensor(np.array(video_mask)), torch.tensor(np.array(words_vec)), \
                torch.tensor(np.array(word_mask)), torch.tensor(np.array(fr_label)), torch.tensor(np.array(scores)), \
                torch.tensor(np.array(scores_mask)), torch.tensor(np.array(id2pos)), torch.tensor(np.array(node_mask)), \
                torch.tensor(np.array(adj_mat)), torch.tensor(np.array(map_gt)), torch.tensor(np.array(duration)), torch.tensor(np.array(gt)), sample_index, \
                torch.tensor(np.array(sub_feats)), torch.tensor(np.array(sub_mask)), torch.tensor(np.array(tef_feats))