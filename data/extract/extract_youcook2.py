import os, sys, glob
import numpy as np
import pandas as pd
from tqdm import tqdm 

feat_csv = glob.glob(os.path.join('.', 'feat_csv', '**', '*.csv'), recursive=True)

for i in tqdm(range(len(feat_csv))) :
    sample = pd.read_csv(feat_csv[i], header=None)
    print(feat_csv[i], sample.values.shape)

    base_folder = '/'.join(os.path.dirname(feat_csv[i]).split('/')[2:])
    des_folder = os.path.join('.', 'feat_npy', base_folder)

    if not os.path.exists(des_folder):
        os.makedirs(des_folder)

    np.save(os.path.join(des_folder, os.path.basename(feat_csv[i])), sample.values)