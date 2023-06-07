import torch 
from tqdm import tqdm
import numpy as np
from collections import OrderedDict, defaultdict

from modules.reloclnet.utils import get_match_labels 
    
def get_svmr_res_from_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs, args, query_metas=None, video2idx=None):
    """
    Args:
        svmr_gt_st_probs: np.ndarray (N_queries, L, L), value range [0, 1]
        svmr_gt_ed_probs:
        query_metas:
        video2idx:
        clip_length: float, how long each clip is in seconds
        min_pred_l: int, minimum number of clips
        max_pred_l: int, maximum number of clips
        max_before_nms: get top-max_before_nms predictions for each query
    
    ->  svmr_gt_st_probs, svmr_gt_ed_probs, args
        
    Returns:
     
    svmr_res: list(dict), each dict is
        {
            "desc": str,
            "desc_id": int,
            "predictions": list(sublist)  # each sublist is
                [video_idx (int), st (float), ed(float), score (float)], video_idx is the same.
         }
    -> list(dict), each dict is
        {
            "predictions": list(sublist), 
            ex. [[ st (float), ed(float), score (float) ], 
                [ st (float), ed(float), score (float) ], ...]
        }
    
    """
    clip_length = args.clip_length
    min_pred_l = args.min_pred_l
    max_pred_l = args.max_pred_l
    max_before_nms = args.max_before_nms
    top_n = args.top_n
    
    svmr_res = []
    # query_vid_names = [e["vid_name"] for e in query_metas]
    # masking very long ones! Since most are relatively short.
    st_ed_prob_product = np.einsum("bm,bn->bmn", svmr_gt_st_probs, svmr_gt_ed_probs)  # (N, L, L)
    # print("st_ed_prob_product :\n", st_ed_prob_product[0])
    valid_prob_mask = generate_min_max_length_mask(st_ed_prob_product.shape, min_l=min_pred_l, max_l=max_pred_l)
    st_ed_prob_product *= valid_prob_mask  # invalid location will become zero!
    batched_sorted_triples = find_max_triples_from_upper_triangle_product(st_ed_prob_product, top_n=max_before_nms,
                                                                          prob_thd=None)
    # print("batched_sorted_triples :\n", batched_sorted_triples[0])
    # """ len(query_vid_names) == batch_size """
    
    for i in range(len(svmr_gt_st_probs)):
    # for i, q_vid_name in tqdm(enumerate(query_vid_names), desc="[SVMR] Loop over queries to generate predictions",
    #                           total=len(query_vid_names)):  # i is query_id
        # q_m = query_metas[i]
        # video_idx = video2idx[q_vid_name]
        _sorted_triples = batched_sorted_triples[i]
        # _sorted_triples[:, 1] += 1  # as we redefined ed_idx, which is inside the moment.
        _sorted_triples[:, :2] = _sorted_triples[:, :2] * clip_length
        # [video_idx(int), st(float), ed(float), score(float)]
        # cur_ranked_predictions = [[video_idx, ] + row for row in _sorted_triples.tolist()]        
        # cur_query_pred = dict(desc_id=q_m["desc_id"], desc=q_m["desc"], predictions=cur_ranked_predictions)
        cur_ranked_predictions = [row[:top_n] for row in _sorted_triples.tolist()]
        # print("cur_ranked_predictions : ", cur_ranked_predictions)
        cur_query_pred = dict(predictions=cur_ranked_predictions)
        svmr_res.append(cur_ranked_predictions)
        
    return svmr_res

def generate_min_max_length_mask(array_shape, min_l, max_l):
    """ The last two dimension denotes matrix of upper-triangle with upper-right corner masked,
    below is the case for 4x4.
    [[0, 1, 1, 0],
     [0, 0, 1, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]
    Args:
        array_shape: np.shape??? The last two dimensions should be the same
        min_l: int, minimum length of predicted span
        max_l: int, maximum length of predicted span
    Returns:
    """
    single_dims = (1, ) * (len(array_shape) - 2)
    mask_shape = single_dims + array_shape[-2:]
    extra_length_mask_array = np.ones(mask_shape, dtype=np.float32)  # (1, ..., 1, L, L)
    mask_triu = np.triu(extra_length_mask_array, k=min_l)
    mask_triu_reversed = 1 - np.triu(extra_length_mask_array, k=max_l)
    final_prob_mask = mask_triu * mask_triu_reversed
    return final_prob_mask  # with valid bit to be 1


def find_max_triples_from_upper_triangle_product(upper_product, top_n=5, prob_thd=None):
    """ Find a list of (k1, k2) where k1 < k2 with the maximum values of p1[k1] * p2[k2]
    Args:
        upper_product (torch.Tensor or np.ndarray): (N, L, L), the lower part becomes zeros, end_idx > start_idx
        top_n (int): return topN pairs with highest values
        prob_thd (float or None):
    Returns:
        batched_sorted_triple: N * [(st_idx, ed_idx, confidence), ...]
    """
    batched_sorted_triple = []
    for idx, e in enumerate(upper_product):
        sorted_triple = top_n_array_2d(e, top_n=top_n)
        if prob_thd is not None:
            sorted_triple = sorted_triple[sorted_triple[2] >= prob_thd]
        batched_sorted_triple.append(sorted_triple)
    return batched_sorted_triple


def top_n_array_2d(array_2d, top_n):
    """ Get topN indices and values of a 2d array, return a tuple of indices and their values,
    ranked by the value
    """
    row_indices, column_indices = np.unravel_index(np.argsort(array_2d, axis=None), array_2d.shape)
    row_indices = row_indices[::-1][:top_n]
    column_indices = column_indices[::-1][:top_n]
    sorted_values = array_2d[row_indices, column_indices]
    return np.stack([row_indices, column_indices, sorted_values], axis=1)  # (N, 3)

    
def eval_by_task_type(moment_predictions, ground_truth,
                     iou_thds=(0.5, 0.7), recall_topks=(1, 5, 10, 100),
                     task_type="SVMR", max_pred_per_query=100, match_number=False, verbose=False, use_desc_type=False):
    
    """ a predicted triplet is positive only if:
    1) its vid_name matches the GT vid_name
    2) IoU between its timestamp and GT timestamp is higher than the given threshold
    moment_predictions w.r.t. different task_type:
        For each query, evaluated on top max_pred_per_query [vid_name, st, ed] triplets. (score entry ignored)
        VCMR: vid_name might be repeating.
        SVMR: vid_name is fixed to be the GT vid_name.
        VR: vid_name is not repeating, st and ed will not be used.
    Args:
        video2idx: {vid_name (str): index (int), ...}
        moment_predictions: list(dict), each dict is {
            "desc": str,
            "desc_id": int,
            "predictions": [vid_name_idx (int), st (float), ed (float), score (float)] * n_pred,
                sorted predictions, n_pred could be different for all dicts. For each prediction,
                only the first 3 elements [vid_name (str), st (float), ed (float),] are used,
                any other following elements are ignored. We leave score here for record.
        }
        ground_truth: list(dict), each dict is {
            "desc": str,
            "desc_id": int,
            "type": str, one of [v, t, vt]
            "vid_name": str
            "ts": [st (float), ed (float)], or list([st (float), ed (float)]), len == 4.
            ...
        }
        iou_thds: temporal IoU thresholds
        recall_topks: recall at different top k
        task_type: str, could be: ["VCMR", "SVMR", "VR"], see TASK_TYPES for definition.
        max_pred_per_query: int, only top max_pred_per_query predictions for each query are used.
        match_number: bool, must set to True if when do evaluation, False is only used for debug.
        verbose:
        use_desc_type: only TVR has desc type
    Returns:
    """
    
    # assert task_type in TASK_TYPES, "task_type must be one of {}".format(list(TASK_TYPES.keys()))
    if verbose:
        print("Running evaluation with task_type {}, n results {}; n gt {}"
              .format(task_type, len(moment_predictions), len(ground_truth)))

    # predictions_by_desc_id = {e["desc_id"]: e for e in moment_predictions}
    # gt_by_desc_id = {e["desc_id"]: e for e in ground_truth}
    desc_type2idx = {"v": 0, "t": 1, "vt": 2}
    desc_types = []  # n_desc

    # if match_number:
    #     assert set(gt_by_desc_id.keys()) == set(predictions_by_desc_id.keys()), \
            # "desc_ids in predictions and ground_truth must match"

    
    pred_info_matrix_collection = []
    # for k, gt_item in tqdm(gt_by_desc_id.items(), desc="Loop over moments", leave=False):
    for k, gt_item in tqdm(enumerate(ground_truth)):
        # if not match_number and k not in predictions_by_desc_id:
        #     continue
        pred_info_matrix = np.array(
            [e[:2] for e in moment_predictions[k]][:max_pred_per_query],
            dtype=np.float32)  # (n_pred, 3) [vid_idx, start, end] -> (n_pred, 2) [start, end]
        # if use_desc_type:
        #     desc_types.append(desc_type2idx[gt_item["type"]])
        # vid_name_matched_pred = pred_info_matrix[:, 0] == video2idx[gt_item["vid_name"]]  # bool, (n_pred, )
        # pred_info_matrix = np.concatenate([pred_info_matrix, vid_name_matched_pred[:, None]], axis=1)  # (n_pred, 4) [vid_idx, start, end, score, vid_name] 

        # add 1 + len(iou_thds) columns, iou_scores, iou_corrects for each iou_thd.
        iou_thd_corrects_columns = []
        if len(gt_item) >= 4:  # didemo, fro all 3 splits, at least 4 ts for each, < 0.5% has more than 4.
            least_n_overlap = 2  # True if overlapped with at least least_n_overlap GT ts.
            iou_corrects_dict = defaultdict(list)
            for single_gt_ts in gt_item:
                single_gt_ts = np.array(single_gt_ts, dtype=np.float32)  # (2, )
                # iou scores of the predictions that have wrong vid_name are set to 0.
                iou_scores = compute_temporal_iou_batch(pred_info_matrix[:, :2], single_gt_ts)
                for iou_thd in iou_thds:
                    iou_corrects_dict[iou_thd].append(iou_scores >= iou_thd)
            for iou_thd in iou_thds:
                iou_corrects = sum(iou_corrects_dict[iou_thd]) >= least_n_overlap  # bool, (n_pred, )
                iou_thd_corrects_columns.append(iou_corrects[:, None])

        else:  # should be 2, len([st, ed]) == 2
            single_gt_ts = np.array(gt_item, dtype=np.float32)  # (2, )
            # iou scores of the predictions that have wrong vid_name are set to 0.
            iou_scores = compute_temporal_iou_batch(pred_info_matrix[:, :2], single_gt_ts)

            for iou_thd in iou_thds:
                iou_corrects = iou_scores >= iou_thd  # bool, (n_pred, )
                iou_thd_corrects_columns.append(iou_corrects[:, None])
                iou_thd_corrects_columns.append(iou_scores[:, None])

        pred_info_matrix = np.concatenate([pred_info_matrix, ] + iou_thd_corrects_columns, axis=1)  # (n_pred, 6) -> (n_pred, 2+ # of ious + mIoU)
        pred_info_matrix_collection.append(pred_info_matrix)

    # column header [vid_name_idx (int), st (float), ed (float), is_vid_name_match (bool),
    # iou_scores>=iou_thd0 (bool), iou_scores>=iou_thd1 (bool)]
    # -> [st(float), end(float), iou_scores>=iou_thd0 (bool), iou_scores>=iou_thd1 (bool), ...]
    # print("pred_info_matrix_collection  before pad :", len(pred_info_matrix_collection),len(pred_info_matrix_collection[0])) # (batch_size, n_pred)
    pred_info_matrix_collection = pad_sequences_1d_np(pred_info_matrix_collection)[0]  # (n_desc, n_pred, 6) -> (n_desc, n_pred, 2 + # of ious + mIoU)
    if use_desc_type:
        desc_types = np.array(desc_types)  # (n_desc)
    # print("pred_info_matrix_collection after pad :", pred_info_matrix_collection.shape) # [batch_size, n_pred, 2+ # of ious]

    # results wrapper
    metrics = OrderedDict()
    metrics_by_type = OrderedDict()

    iou_c_offset = 2  # iou_corrects column index starts here
    
    # SVMR
    # vid_name_matched = pred_info_matrix_collection[:, :, 3].astype(bool)  # (n_desc, n_pred)
    # n_desc = len(vid_name_matched)
    n_desc = len(pred_info_matrix_collection) # batch_size
    for iou_idx, iou_thd in enumerate(iou_thds):
        iou_corrects = pred_info_matrix_collection[:, :, iou_c_offset + iou_idx].astype(bool)  # (n_desc, n_pred)
        # 1) there might be more than one positive clip, so use `>= 1`
        for k in recall_topks:
            metrics["{}-r{}".format(iou_thd, k)] = get_rounded_percentage(np.mean(
                [np.sum(iou_corrects[idx][:k]) >= 1 for idx in range(n_desc)]
            ))
    iou_corrects = pred_info_matrix_collection[:,:,-1]   
    metrics["mIoU"] = get_rounded_percentage(np.mean([iou_corrects[idx][:1] for idx in range(n_desc)]))
    
    if use_desc_type:
        for desc_type in desc_type2idx:
            type_corrects = desc_types == desc_type2idx[desc_type]  # (n_desc)
            n_desc_in_type = np.sum(type_corrects)  # (n_desc)
            for iou_idx, iou_thd in enumerate(iou_thds):
                # (n_desc, n_pred)
                iou_corrects = pred_info_matrix_collection[:, :, iou_c_offset + iou_idx].astype(bool)
                # 1) there might be more than one positive clip, so use `>= 1`
                for k in recall_topks:
                    metrics_by_type["{}-{}-r{}".format(desc_type, iou_thd, k)] = get_rounded_percentage(
                        1.0 * np.sum([np.sum(iou_corrects[idx][:k]) >= 1 and type_corrects[idx]
                                        for idx in range(n_desc)])
                        / n_desc_in_type)

    if use_desc_type:
        metrics_by_type["desc_type_ratio"] = "v {} t {} vt {}"\
            .format(*[get_rounded_percentage(1.0 * np.sum(desc_types == desc_type2idx[k]) / len(desc_types))
                      for k in ["v", "t", "vt"]])
    return metrics, metrics_by_type

def get_rounded_percentage(float_number, n_floats=2):
    return round(float_number * 100, n_floats)

def pad_sequences_1d_np(sequences, dtype=np.float32):

    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        dtype: np.dtype or torch.dtype
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,4), np.random.randn(4,3,4), np.random.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)
    """
    if isinstance(sequences[0], list):
        sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    assert "numpy" in str(dtype), "dtype and input type does not match"
    padded_seqs = np.zeros((len(sequences), max(lengths)) + extra_dims, dtype=dtype)
    mask = np.zeros((len(sequences), max(lengths)), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask


def compute_temporal_iou_batch(preds, gt):
    """ compute intersection-over-union along temporal axis
    This function is significantly faster than `compute_temporal_iou`,
    the result should be the same.
    Args:
        preds: np.ndarray, (N, 2), [st (float), ed (float)] * N
        gt: [st (float), ed (float)]
    Returns:
        iou (float): np.ndarray, (N, )
    References:
        for np.divide with zeros, see https://stackoverflow.com/a/37977222
    """
    intersection = np.maximum(0, np.minimum(preds[:, 1], gt[1]) - np.maximum(preds[:, 0], gt[0]))
    union = np.maximum(preds[:, 1], gt[1]) - np.minimum(preds[:, 0], gt[0])  # not the correct union though
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)




def cat_tensor(tensor_list):
    if len(tensor_list) == 0:
            return None
    else:
        seq_l = [e.shape[1] for e in tensor_list]
        b_sizes = [e.shape[0] for e in tensor_list]
        b_sizes_cumsum = np.cumsum([0] + b_sizes)
        if len(tensor_list[0].shape) == 3:
            hsz = tensor_list[0].shape[2]
            res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
        elif len(tensor_list[0].shape) == 2:
            res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
        else:
            raise ValueError("Only support 2/3 dimensional tensors")
        for i, e in enumerate(tensor_list):
            res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i+1], :seq_l[i]] = e
        return res_tensor

def compute_context_info(model, data_loader, args):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated 2200 (videos) * 100 (frm) * 500 (hsz) * 4 (B) * 2 (video/sub) * 2 (layers) / (1024 ** 2) = 1.76 GB
    max_n_videos: only consider max_n_videos videos for each query to return st_ed scores.
    """
    video_feats, video_masks, sub_feats, sub_masks = [[] for _ in range(4)]
    
    for bid, batch in enumerate(data_loader, 1):
        index, vid_feat, video_mask, words_vec, word_mask, label, \
        scores, scores_mask, id2pos, node_mask, adj_mat, map_gt, duration, gt, sample_index, \
        sub_feat, sub_mask = batch 
                        
        match_label = torch.tensor(get_match_labels(label, args.max_num_frames))
        model_input = {'query_feat' : words_vec.cuda(), 'query_mask': word_mask.cuda(), \
                'video_feat':vid_feat.cuda(), 'video_mask':video_mask.cuda(), \
                'sub_feat':sub_feat.cuda(), 'sub_mask': sub_mask.cuda(), \
                'st_ed_indices': label.cuda(), 'match_labels': match_label.cuda()}
                        
        _video_feat, _sub_feat = model.encode_context(model_input["video_feat"], model_input["video_mask"],\
                                                model_input["sub_feat"], model_input["sub_mask"])
        video_feats.append(_video_feat)
        video_masks.append(model_input["video_mask"])
        sub_feats.append(_sub_feat)
        sub_masks.append(model_input["sub_mask"])
    
    
    return dict(
                video_feat=cat_tensor(video_feats),  # (N_videos, L, hsz),
                video_mask=cat_tensor(video_masks),  # (N_videos, L)
                sub_feat=cat_tensor(sub_feats),
                sub_mask=cat_tensor(sub_masks))
