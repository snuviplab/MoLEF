import numpy as np
import torch
from datasets.Activitynet import Activitynet
from datasets.Tacos import Tacos
from datasets.Charades import Charades
from datasets.Didemo import Didemo
from datasets.Youcook2 import Youcook2
from datasets.Tvr import Tvr
from datasets.Msrvtt import Msrvtt
from utils.utils import load_json, generate_anchors
from utils import metric
import importlib

# activitynet, tacos, charades, didemo, Tvr

def get_dataset(dataset, model_name, feature_path, data_path, word2vec, max_num_frames, max_num_words, max_num_nodes,
                is_training=True, is_adj=True, with_max_IoU=False):

    get_class  = getattr(getattr(importlib.import_module("datasets"), dataset), dataset)

    if dataset in ['Activitynet', 'Tacos', 'Charades', 'Didemo', 'Tvr', 'Youcook2', 'Msrvtt']:
        return get_class(model_name, feature_path, data_path, word2vec, max_num_frames, max_num_words, max_num_nodes, is_training, is_adj, with_max_IoU)
    else : 
        return None

if __name__ == "__main__" :
    from utils.utils import load_json, generate_anchors, load_word2vec
    from dataloaders.dataloader import get_dataset

    dataset = "Activitynet"
    model_name = 'cmin'
    feature_path = "/data/projects/VT_localization/tsgv_data/data/activity/org"
    data_path = "/data/projects/VT_localization/tsgv_data/data/activity/test_data.json"
    word2vec = load_word2vec("/data/projects/VT_localization/tsgv_data/data/glove.840B.300d.bin")
    print("done")
    max_num_frames = 200
    max_num_words = 20
    max_num_nodes = 20
    is_training = False
    is_adj=False  ## True if model is ianet
    with_max_IoU=-float('inf') ## 0.3 if model is cmin
    train_loader = get_dataset(dataset, model_name, feature_path, data_path, word2vec, max_num_frames, max_num_words, max_num_nodes, is_training, is_adj, with_max_IoU)
    print(len(train_loader))
    it = iter(train_loader)
    
    ind, vid_feats, video_mask, words_vec, word_mask, fr_label, \
    scores, scores_mask, id2pos, node_mask, adj_mat, map_gt, duration, gt, sample_index, \
    raw_vid_feats, raw_words_vec, loc_start, loc_end, localization, raw_vid_feats_length, factors = next(it)
    # sub_feats, sub_mask, tef_feats
    print(len(next(it)))
    print("done")
    # print(first)
