import os
import numpy as np
import h5py

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--feature_dir", type=str, required=True, help="feature path", default='tvr_feature_release')
parser.add_argument("--save_dir", type=str, required=False, help="save dir", default='data/tvr/features')
args = parser.parse_args()

query_types = ['query_only', 'query_w_sub', 'sub_query']
query_fpath = ['bert_feature/query_only/tvr_query_pretrained_w_query.h5', 
               'bert_feature/sub_query/tvr_query_pretrained_w_sub_query.h5',
               'bert_feature/sub_query/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5']

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    os.makedirs(os.path.join(args.save_dir, "video_feature"))
    for qtype in query_types :
        os.makedirs(os.path.join(args.save_dir, qtype))

## video feature
print(f"Loading video_feature data ...")
fdata = h5py.File(os.path.join(args.feature_dir, "video_feature/tvr_resnet152_rgb_max_i3d_rgb600_avg_cat_cl-1.5.h5")) # tvr_i3d_rgb600_avg_cl-1.5.h5 / tvr_resnet152_rgb_max_cl-1.5.h5
keys = fdata.keys()
for key in keys :
    ftr = fdata[key][:,:]
    np.save(os.path.join(args.save_dir, f'video_feature/{key}.npy'), ftr)
    # break

## query/subtitle feature
for qtype, qpath in list(zip(query_types, query_fpath)):
    print(f"Loading {qtype} data ...")
    fdata = h5py.File(os.path.join(args.feature_dir, qpath), "r")
    keys = fdata.keys()
    for key in keys :
        print(f"Converting desc_id = {key}...")
        ftr = fdata[key][:,:]
        np.save(os.path.join(args.save_dir, f'{qtype}/{key}.npy'), ftr)
        # break
    
