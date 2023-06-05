import torch
import numpy as np
from utils import pad_sequences_1d


def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]
    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if "feat" in k:
            if k in ['video_feat', 'sub_feat', 'tef_feat']:
                fixed_length = 128
            else:
                fixed_length = None
            batched_data[k] = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=torch.float32,
                                               fixed_length=fixed_length)
    fixed_length = 128
    if "st_ed_indices" in model_inputs_keys:
        st_ed_indices = [e["model_inputs"]["st_ed_indices"] for e in batch]
        # construct moment localization labels
        batched_data["st_ed_indices"] = torch.stack(st_ed_indices, dim=0)
        # construct moment localization foreground and background labels
        match_labels = np.zeros(shape=(len(st_ed_indices), fixed_length), dtype=np.int32)
        for idx, st_ed_index in enumerate(st_ed_indices):
            st_ed = st_ed_index.cpu().numpy()
            st, ed = st_ed[0], st_ed[1]
            match_labels[idx][st:(ed + 1)] = 1
        batched_data['match_labels'] = torch.tensor(match_labels, dtype=torch.long)
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = {}
    for k, v in batched_model_inputs.items():
        if "feat" in k:
            model_inputs[k] = v[0].to(device, non_blocking=non_blocking)
            model_inputs[k.replace("feat", "mask")] = v[1].to(device, non_blocking=non_blocking)
        else:
            model_inputs[k] = v.to(device, non_blocking=non_blocking)
    return model_inputs