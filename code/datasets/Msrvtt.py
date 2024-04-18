import os
import copy
import numpy as np
import math
import skimage.measure as scikit # for tga
import torch
from torch.utils.data import Dataset
from utils.utils import load_feature, tokenize, load_json, generate_anchors
from utils.io_utils import load_pkl
import nltk # for cpl
from utils import metric
from modules.tdtan.core.eval import iou

# for trm
from modules.trm.data.datasets.utils import moment_to_iou2d, bert_embedding, get_vid_feat
from transformers import DistilBertTokenizer

class Msrvtt(Dataset):
    def __init__(self, model_name, feature_path, data_path, word2vec, max_num_frames, max_num_words, max_num_nodes, is_training, is_adj, vocab_size, with_max_IoU=-float('inf')):
        self.model_name = model_name
        self.feature_path = feature_path
        self.data_path = data_path
        self.word2vec = word2vec
        self.max_num_frames = max_num_frames
        self.max_num_words = max_num_words
        self.max_num_nodes = max_num_nodes
        self.is_training = is_training
        self.is_adj = is_adj
        self.with_max_IoU = with_max_IoU
        self.epsilon = 1E-10
        
        self.data = load_json(self.data_path)
        self.data_phrase = load_json(self.data_path.split('.')[0]+'_phrase.json') # for TRM
        self.anchors = generate_anchors(dataset='Msrvtt')
        widths = (self.anchors[:, 1] - self.anchors[:, 0] + 1)  # [num_anchors]
        centers = np.arange(0, max_num_frames)  # [video_len]
        start = np.expand_dims(centers, 1) - 0.5 * (np.expand_dims(widths, 0) - 1)
        end = np.expand_dims(centers, 1) + 0.5 * (np.expand_dims(widths, 0) - 1)
        self.proposals = np.stack([start, end], -1)  # [video_len, num_anchors, 2]

        # cal words id 
        self.vocab = load_pkl(file_path="path/to/data/msrvtt/msr_glove.pkl")
        self.keep_vocab = dict()
        for w, _ in self.vocab['counter'].most_common(vocab_size) : 
            self.keep_vocab[w] = len(self.keep_vocab) + 1

        # for TRM, using DistillBERT4
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        
    @property
    def vocab_size(self) : 
        return len(self.keep_vocab) + 1
    
    def __getitem__(self, ind):
        vid, duration, timestamps, sentence, words, id2pos, adj_mat, phrase = self.data_phrase[ind]
        adj_mat = np.asarray(adj_mat).astype(np.int32)

        # do not shuffle the comments order: video feature -> frame label computing -> ...
        # video feature
        # print("VID", vid)
        vid_feats = load_feature(os.path.join(self.feature_path, '%s.npy'%vid), dataset='Msrvtt')
        
        # for tmlga
        raw_vid_feats = copy.deepcopy(vid_feats)
        raw_vid_feats_length = raw_vid_feats.shape[0]

        # frame label computing 
        fps = vid_feats.shape[0] / duration
        adj_mat = np.asarray(adj_mat)
        start_frame = int(fps * timestamps[0])
        end_frame = int(fps * timestamps[1])
        if end_frame >= vid_feats.shape[0]:
            end_frame = vid_feats.shape[0] - 1
        if start_frame > end_frame:
            start_frame = end_frame
        assert start_frame <= end_frame
        assert 0 <= start_frame < vid_feats.shape[0]
        assert 0 <= end_frame < vid_feats.shape[0]
        fr_label = np.asarray([start_frame, end_frame]).astype(np.int32)
        ori_fr_label = copy.deepcopy(fr_label)

        # moment, iou2d for trm
        from modules.trm.config import config
        if timestamps[0] < timestamps[1]:
            moment = torch.Tensor([max(timestamps[0], 0), min(timestamps[1], duration)])
            iou2d = moment_to_iou2d(moment, config.MODEL.TRM.NUM_CLIPS, duration)
        else:
            moment = torch.Tensor([0, 0])
            iou2d = moment_to_iou2d(moment, config.MODEL.TRM.NUM_CLIPS, duration)

        # word embedding 
        words_vec = np.asarray([self.word2vec[word] if word in self.word2vec else np.zeros(300) for word in words ]) 

        words_vec = words_vec.astype(np.float32)
        id2pos = np.asarray(id2pos).astype(np.int64)

        # for tmlga
        raw_words_vec = copy.deepcopy(words_vec)

        # word padding
        ori_words_len = words_vec.shape[0]
        if ori_words_len < self.max_num_words:
            word_mask = np.zeros([self.max_num_words], np.uint8)
            word_mask[range(ori_words_len)] = 1
            words_vec = np.pad(words_vec, ((0, self.max_num_words - ori_words_len), (0, 0)), mode='constant')
        else:
            word_mask = np.ones([self.max_num_words], np.uint8)
            words_vec = words_vec[:self.max_num_words]
        
        # video sampling
        ori_video_len = vid_feats.shape[0]
        video_mask = np.ones([self.max_num_frames], np.uint8)
        sample_index = np.linspace(start=0, stop=ori_video_len - 1, num=self.max_num_frames).astype(np.int32)
        new_video = []
        for i in range(len(sample_index) - 1):
            start = sample_index[i]
            end = sample_index[i + 1]
            if start == end or start + 1 == end:
                new_video.append(vid_feats[start])
            else:
                new_video.append(np.mean(vid_feats[start: end], 0))
        new_video.append(vid_feats[-1])
        vid_feats = np.stack(new_video, 0)

        # frame label recomputing
        fr_label[0] = min(np.where(sample_index >= fr_label[0])[0])
        if fr_label[1] == vid_feats.shape[0] - 1:
            fr_label[1] = self.max_num_frames - 1
        else:
            fr_label[1] = max(np.where(sample_index <= fr_label[1])[0])
        if fr_label[1] < fr_label[0]:
            fr_label[0] = fr_label[1]
            
        # reloclnet
        sub_feats = vid_feats.copy()
        sub_mask = video_mask.copy()
        tef_feats = np.zeros([self.max_num_frames, 2], dtype=np.uint8)
        
        if self.model_name == 'reloclnet':
            from modules.reloclnet.utils import uniform_feature_sampling, get_st_ed_label
            vid_feats = load_feature(os.path.join(self.feature_path, '%s.npy'%vid), dataset='Msrvtt')
            vid_feats = uniform_feature_sampling(vid_feats, self.max_num_frames)
            ctx_l = len(vid_feats)
            
            tef_st = np.arange(0, ctx_l, 1.0, dtype=np.float32) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef_feats = np.stack([tef_st, tef_ed], axis=1)
            
            if vid_feats.shape[0] < self.max_num_frames :
                video_mask = np.zeros([self.max_num_frames], np.uint8)
                video_mask[:vid_feats.shape[0]] = 1
                vid_feats = np.pad(vid_feats, 
                                ((0,self.max_num_frames-vid_feats.shape[0]), (0,0)))
                tef_feats = np.pad(tef_feats, 
                                ((0,self.max_num_frames-tef_feats.shape[0]), (0,0)))

            sub_feats = vid_feats.copy()
            sub_mask = video_mask.copy()
            
            fr_label = get_st_ed_label(ts=timestamps, max_idx=ctx_l-1, clip_length=5)    

        # tga - video features 
        seg_size = 0
        seg_break = [0, 0, 0]
        if self.model_name == 'tga':
            seg_size = int(fps * 15.13) // 2 # average moment length of MSR-VTT 15.13
            video_feat1 = scikit.block_reduce(raw_vid_feats, block_size=(seg_size, 1), func=np.mean)
            video_feat2 = scikit.block_reduce(raw_vid_feats, block_size=(seg_size*2, 1), func=np.mean)
            video_feat3 = scikit.block_reduce(raw_vid_feats, block_size=(seg_size*4, 1), func=np.mean)
            # concatenation of all frame features
            vid_feats = np.concatenate((video_feat1,video_feat2, video_feat3), axis=0)  
            seg_break = [video_feat1.shape[0], video_feat2.shape[0], video_feat3.shape[0]]

        # check origin dimension 
        # assert len(id2pos) == adj_mat.shape[0] == ori_words_len
        # some words have been cut out
        true_index = id2pos < self.max_num_words
        id2pos = id2pos[true_index]
        adj_mat = adj_mat[true_index]
        adj_mat = adj_mat[:, true_index]
        # for training graph convolution
        # node padding
        if id2pos.shape[0] < self.max_num_nodes:
            node_mask = np.zeros([self.max_num_nodes], np.uint8)
            node_mask[range(id2pos.shape[0])] = 1
            id2pos = np.pad(id2pos, (0, self.max_num_nodes - id2pos.shape[0]), mode='constant')
            adj_mat = np.pad(adj_mat,
                             ((0, self.max_num_nodes - adj_mat.shape[0]),
                              (0, self.max_num_nodes - adj_mat.shape[1])),
                             mode='constant')
        else:
            node_mask = np.ones([self.max_num_nodes], np.uint8)
            id2pos = id2pos[:self.max_num_nodes]
            adj_mat = adj_mat[:self.max_num_nodes, :self.max_num_nodes]
        # scores computing
        proposals = np.reshape(self.proposals, [-1, 2])
        illegal = np.logical_or(proposals[:, 0] < 0, proposals[:, 1] >= self.max_num_frames)
        label1 = np.repeat(np.expand_dims(fr_label, 0), proposals.shape[0], 0)
        IoUs = metric.calculate_IoU_batch((proposals[:, 0], proposals[:, 1]),
                                            (label1[:, 0], label1[:, 1]))
        IoUs[illegal] = 0.0  # [video_len * num_anchors]
        max_IoU = np.max(IoUs)
        if self.model_name in ['cmin', 'csmgan', 'ianet'] and max_IoU == 0.0:
            print(illegal)
            print(fr_label)
            print(proposals[illegal])
            print(proposals[1 - illegal])
            # print(IoUs)
            # print(label, max_IoU)
            exit(1)
        
        ## ianet, csmgan
        if self.is_adj != False:  #  ianet, csmgan use this setting
            adj_mat = copy.deepcopy(IoUs)
            adj_mat[adj_mat < 0.8 * max_IoU] = 0.
        ## cmin    
        IoUs[IoUs < self.with_max_IoU * max_IoU] = 0.0  # cmin uses with_max_IoU as 0.3
        IoUs = IoUs / max_IoU
        scores = IoUs.astype(np.float32)
        scores_mask = (1 - illegal).astype(np.uint8)
        ## tdtan, ranet
        # To do : num_sample_clips == max_num_frames, change tdtan config
        if self.model_name == 'tdtan' :
            from modules.tdtan.model_config import config
            num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
        elif self.model_name == 'ranet' :
            from modules.ranet.model_config import config 
            num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
        elif self.model_name == 'mgpn' :
            from modules.mgpn.model_config import config 
            num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
        else :
            num_clips = self.max_num_frames
        s_times = np.arange(0,num_clips, dtype=np.float32)*duration/num_clips
        e_times = np.arange(1,num_clips+1, dtype=np.float32)*duration/num_clips
        s_len = s_times[:, None].shape[0]
        e_len = e_times[None, :].shape[1]
        map_gt = iou(np.stack([np.broadcast_to(s_times[:,None], (s_len, num_clips)),
                                    np.broadcast_to(e_times[None,:], (num_clips, e_len))], axis=2).reshape(-1,2).tolist(),
                    np.array(timestamps).tolist()).reshape(num_clips, num_clips)            
        map_gt[np.isnan(map_gt)] = 0  # for avoiding nan value, fill 0 
        gt = np.array(timestamps)
        ## tmlga
        localization = np.zeros(raw_vid_feats_length, dtype=np.float32)
        tmlga_start = math.floor(ori_fr_label[0]) # feature start
        tmlga_end = math.floor(ori_fr_label[1])
        
        loc_start = np.ones(raw_vid_feats_length, dtype=np.float32) * self.epsilon
        loc_end = np.ones(raw_vid_feats_length, dtype=np.float32) * self.epsilon
        y = (1 - (raw_vid_feats_length-3) * self.epsilon - 0.5)/ 2
        if tmlga_start > 0:
            loc_start[tmlga_start - 1] = y
        if tmlga_start < raw_vid_feats_length-1:
            loc_start[tmlga_start + 1] = y
        loc_start[tmlga_start] = 0.5
        if tmlga_end > 0:
            loc_end[tmlga_end - 1] = y
        if tmlga_end < raw_vid_feats_length-1:
            loc_end[tmlga_end + 1] = y
        loc_end[tmlga_end] = 0.5
        y = 1.0
        localization[tmlga_start:tmlga_end] = y
        localization = torch.from_numpy(localization)
        loc_start = torch.from_numpy(loc_start)
        loc_end = torch.from_numpy(loc_end)
        factors = duration/raw_vid_feats_length

        # cnm 
        words_id = np.asarray([self.keep_vocab[word] for word in words if word in self.keep_vocab])        
        words_id = np.expand_dims(words_id, 1)

        # word padding
        if words_id.shape[0] < self.max_num_words:
            words_id = np.pad(words_id, ((0, self.max_num_words - words_id.shape[0]), (0, 0)), mode='constant', constant_values = 0)
        else:
            words_id = words_id[:self.max_num_words]

        if self.model_name == 'msdetr':
            words_id = np.asarray([self.keep_vocab.get(word, len(self.keep_vocab) + 1) for word in words if word in self.word2vec])
            words_id = np.expand_dims(words_id, 1)
            sub_mask = np.array([np.random.uniform() < 0.15 for _ in range(words_id.shape[0])])
            if np.sum(sub_mask) == 0 or np.sum(sub_mask) == words_id.shape[0]:
                random_idx = np.random.choice(np.arange(words_id.shape[0]))
                sub_mask[random_idx] ^= 1
            sub_mask = np.expand_dims(sub_mask, 1)
            if len(words) < self.max_num_words:
                sub_mask = np.pad(sub_mask, ((0, self.max_num_words - len(words)), (0, 0)), mode='constant')
            else:
                sub_mask = sub_mask[:self.max_num_words]
            # word padding
            if len(words) < self.max_num_words:
                words_id = np.pad(words_id, ((0, self.max_num_words - len(words)), (0, 0)), mode='constant')
            else:
                words_id = words_id[:self.max_num_words]

        # trm phrases
        new_phrase = []
        for i in range(len(phrase)):
            new_phrase.append(phrase[i])
        if len(new_phrase) == 0:
            new_phrase.append(sentence)

        queries, word_lens = bert_embedding(sentence, self.tokenizer)  # padded query of N*word_len, tensor of size = N
        
        # for cpl and cnm
        weights = np.zeros((1,1)) # Probabilities to be masked
        valid_words_length = 0
        valid_vid_feats_length = 0
        if self.model_name in ['cpl', 'cnm']:
            weights = []
            words = []
            for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
                if word in self.keep_vocab:
                    if 'NN' in tag:
                        weights.append(2)
                    elif 'VB' in tag:
                        weights.append(2)
                    elif 'JJ' in tag or 'RB' in tag:
                        weights.append(2)
                    else:
                        weights.append(1)
                    words.append(word)
            valid_words_length = min(len(weights), self.max_num_words)
            # padding
            tmp = np.exp(weights[:valid_words_length] - np.max(weights))
            sum_tmp = np.sum(tmp)
            weights[:valid_words_length] = [t/sum_tmp for t in tmp]
            weights = np.expand_dims(weights, 1)
            if len(weights) < self.max_num_words:
                weights = np.pad(weights, ((0, self.max_num_words - valid_words_length), (0, 0)), mode='constant', constant_values=0).astype(np.float32)
            else:
                weights = weights[:self.max_num_words].astype(np.float32)

            valid_vid_feats_length = min(len(vid_feats), self.max_num_frames)
        words_feat = words_vec.copy()
        words_feat = np.concatenate([np.zeros((1,300)), words_vec], axis = 0).astype(np.float32)      
        words_feat[-1] = np.zeros((1,300))

        # vlsnet
        start_idx = np.argmax(map_gt) // num_clips
        end_idx = np.argmax(map_gt) % num_clips
        h_labels = np.zeros(self.max_num_frames, dtype=np.int32)
        extend = 0.1
        cur_max_len = raw_vid_feats_length if raw_vid_feats_length < self.max_num_frames else self.max_num_frames
        extend_len = round(extend * float(end_idx - start_idx + 1))
        if extend_len > 0:
            st_ = max(0, start_idx - extend_len)
            et_ = min(end_idx + extend_len, cur_max_len - 1)
            h_labels[st_:(et_ + 1)] = 1
        else:
            h_labels[start_idx:(end_idx + 1)] = 1

        return ind, vid_feats, video_mask, words_vec, word_mask, fr_label, \
            scores, scores_mask, id2pos, node_mask, adj_mat, map_gt, duration, gt, sample_index, \
            raw_vid_feats, raw_words_vec, loc_start, loc_end, localization, raw_vid_feats_length, factors, \
            sub_feats, sub_mask, tef_feats, words_id, seg_size, ori_fr_label, seg_break, \
            sentence, new_phrase, moment, iou2d, queries, word_lens, valid_words_length, weights, words_feat, valid_vid_feats_length, \
            start_idx, end_idx, h_labels
            
    def __len__(self):
        return len(self.data)
