import os, sys, glob
import numpy as np
import h5py

if __name__ == '__main__' : 

    parser = ArgumentParser()
    parser.add_argument("--feature_dir", type=str, required=True, help="feature path", default='tvr_feature_release')
    parser.add_argument("--save_dir", type=str, required=False, help="save dir", default='data/tvr/features')
    args = parser.parse_args()

    data = glob.glob(os.path.join(args.feature_dir, '**', '*.h5'), recursive=True)
    print('The number of data', len(data))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i in range(len(data)) :
        basename = os.path.basename(data[i])
        filename = os.path.splitext(basename)[0]

        with h5py.File(data[i], "r") as f:
            X = np.asarray(f['features'])
        
        save_path  = os.path.join(args.save_dir, filename)
        np.save(save_path, X)

        print("%s Done" % filename)

