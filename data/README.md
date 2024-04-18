# Prepare dataset and features
- The directory of each dataset includes train_data.json, val_data.json, test_data.json, {dataset}_glove.pkl and feature folder.
```
├── data 
    ├── activitynet
    │    ├── feature
    │    ├── train_data.json
    │    ├── val_data.json
    │    └── test_data.json
    ├── charades
    │    ├── feature 
    │    ├── train_data.json
    │    └── ...
    ├── didemo
    │    ├── feature
    │    ├── train_data.json
    │    └── ...
    └── tacos
        ├── feature
        ├── train_data.json
        └── ...
```
- For the DiDeMo and TVR datasets, you need to request the train_data.json files to us due to the memory buffer issue on GitHub. Or download the files from [here](https://drive.google.com/drive/folders/1VKdX1DHYRrHYjLH91vl5rn86N6MrubzE?usp=sharing)
- Each dataset needs the vocab file ({dataset}_glove.pkl), download the files from [here](https://drive.google.com/drive/folders/1VKdX1DHYRrHYjLH91vl5rn86N6MrubzE?usp=sharing) and modify the file path in datasets/{dataset}.py 
- The features must be in `npy` format to utilize our framework. 
- We use the pre-trained 3D ConvNets ([here](https://github.com/piergiaj/pytorch-i3d)) to prepare the visual features, the 
extraction codes are placed in this folder. Please download the pre-trained weights [`rgb_charades.pt`](
https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_charades.pt) and [`rgb_imagenet.pt`](
https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt). 
- The pre-trained GloVe and RoBERTa embeddings are available at [link](https://drive.google.com/file/d/1XOlwnO2lMeqio8A6pxHzPs3La0eD6sWk/view?usp=sharing).
- Some instruction details are copied from [[26hzhang/VSLNet]](https://github.com/26hzhang/VSLNet/tree/master/prepare) and revised appropriately for our experiments. 

## ActivityNet Captions 
- The train/test sets of ActivityNet Caption are available at [here](
https://cs.stanford.edu/people/ranjaykrishna/densevid/).
- We have the codes to convert the C3D visual features provided in [ActivityNet official website](
http://activity-net.org/challenges/2016/download.html)
- Download the C3D visual features
```shell script
mkidr data/activitynet/features && cd "$_"
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-00
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-01
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-02
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-03
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-04
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-05
cat activitynet_v1-3.part-* > features.zip && unzip features.zip
rm features.zip
rm activitynet_v1-3.part-*
```
- Convert the features to `npy` format as
```shell script
python3 extract/extract_activitynet.py --dataset_dir <path to activitynet caption annotation dataset>  \
      --hdf5_file <path to downloaded C3D features>  \
      --save_dir <path to save extracted features>
```
- (Optional) You can download the features from the drive of [[26hzhang/VSLNet]](https://app.box.com/s/h0sxa5klco6qve5ahnz50ly2nksmuedw)

## Charades-STA
- The videos/images for Charades-STA dataset is available at [here](https://prior.allenai.org/projects/charades), please download 
either `RGB frames at 24fps (76 GB)` (image frames) or `Data (original size) (55 GB)` (videos). For the second one, the 
extractor will automatically decompose the video into images.
```shell script
# download RGB frames
wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_rgb.tar
# or, download videos
wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip
```
- Download the features: 
```shell script
# download two-stream features (RGB)
wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_features_rgb.tar.gz

```

## TACoS
- The videos of TACoS is from MPII 
Cooking Composite Activities dataset, which can be download [here](
https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/human-activity-recognition/mpii-cooking-composite-activities/).
- TACoS features are from [[jiyanggao/TALL]](https://github.com/jiyanggao/TALL), while the videos, and the features are converted with the followings: 
```shell script
python extract/extract_tacos.py --data_path <path to tacos annotation dataset>  \
      --feature_path <path to downloaded C3D features>  \
      --save_dir <path to save extracted visual features>  \
      --sample_rate 64  # sliding windows
```

## DiDeMo
- DiDeMo dataset is from [[LisaAnne/LocalizingMoments]](https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md), and the features can be downloaded from [here](https://drive.google.com/drive/u/1/folders/1_oyJ5rQiZboipbMl6tkhY8v0s9zDkvJc)
- Use the code to convert the features to `npy` format: 
```shell script
python extract/extract_didemo.py --feature_dir <path to tvr_feature_release> --save_dir <path to save npy features>
```

## YouCook2
- YouCook2 dataset can be downloaded from [here](http://youcook2.eecs.umich.edu/download)
- If you download the [ResNet features](http://youcook2.eecs.umich.edu/static/YouCookII/feat_csv.tar.gz), you have to convert `csv` format to `npy` for our experiments. Use the code: 
```shell script
python extract/extract_youcook2.py
```

## TVR
- Tvr dataset is from [[jayleicn/TVRetrieval]](https://github.com/jayleicn/TVRetrieval) and the features can be downloaded from [here](https://drive.google.com/file/d/1j4mVkXjKCgafW3ReNjZ2Rk6CKx0Fk_n5/view?usp=sharing). After downloading the feature file, extract it to ` data/tvr/featrues` directory: 
```shell script
mkdir data/tvr/features
tar -xf path/to/tvr_feature_release.tar.gz -C data
```
- You have to convert `h5py` format to `npy` for our experiments. Use the code:
```shell script
python extract/extract_tvr.py --feature_dir <path to tvr_feature_release> --save_dir <path to save npy features>
```
- (Optional) If you have an issue downloading the features, please request the drive link from us.

## MSRVTT
- This dataset was originally intended for a retrieval task, but we created a new dataset for our moment localization task by extracting the original video from the YouTube link provided by the authors.
- The links of original videos are from [here](https://www.mediafire.com/folder/h14iarbs62e7p/shared). 
- We use this [repo](https://github.com/katsura-jp/extruct-video-feature) to extract features. 
