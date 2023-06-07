import torch
import numpy as np
from modules.tmlga.utils import rnns

class BatchCollator(object):

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        '''
        index     = transposed_batch[0]
        videoFeat = transposed_batch[1]
        tokens    = transposed_batch[2]
        start = transposed_batch[3]
        end = transposed_batch[4]
        localiz = transposed_batch[5]
        time_start = transposed_batch[6]
        time_end = transposed_batch[7]
        factor = transposed_batch[8]
        fps = transposed_batch[9]

        videoFeat, videoFeat_lengths = rnns.pad_sequence(videoFeat)
        localiz, localiz_lengths = rnns.pad_sequence(localiz)
        tokens, tokens_lengths   = rnns.pad_sequence(tokens)

        start, start_lengths = rnns.pad_sequence(start)
        end, end_lengths     = rnns.pad_sequence(end)
        '''
        ind = transposed_batch[0]
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
        raw_vid_feats = transposed_batch[15]
        raw_words_vec = transposed_batch[16]
        loc_start = transposed_batch[17]
        loc_end = transposed_batch[18]
        localization = transposed_batch[19]
        raw_vid_feats_length = transposed_batch[20]
        factors = transposed_batch[21]

        videoFeat, videoFeat_lengths = rnns.pad_sequence(raw_vid_feats)
        localiz, localiz_lengths = rnns.pad_sequence(localization)
        tokens, tokens_lengths = rnns.pad_sequence(raw_words_vec)

        start, start_lengths = rnns.pad_sequence(loc_start)
        end, end_lengths = rnns.pad_sequence(loc_end)

        time_start = gt[0]
        time_end = gt[1]
        fps = 1


        return ind, \
               videoFeat, \
               videoFeat_lengths, \
               tokens, \
               tokens_lengths, \
               start,  \
               end, \
               localiz, \
               localiz_lengths, \
               torch.tensor(np.array(gt)), \
               factors, \
               fps
