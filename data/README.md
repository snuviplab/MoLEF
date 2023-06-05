# Prepare data
directory structure

```
├── data 
    ├── activitynet
    │    ├── new 
    │    ├── org
    │    ├── org_pca 
    │    ├── train_data.json
    │    ├── valid_data.json
    │    └── test_data.json
    ├── charades
    │    ├── new 
    │    ├── org
    │    ├── org_pca
    │    └── ...
    ├── didemo
    │    ├── org
    │    ├── org_pca
    │    └── ...
    └── tacos
        ├── new
        ├── org
        ├── org_pca
        └── ...
```

- Download glove 300d file from [link](https://drive.google.com/file/d/1XOlwnO2lMeqio8A6pxHzPs3La0eD6sWk/view?usp=sharing) and should be placed in `data/`. 
- For the feature of dataset, see viplab server /data/projects/VT_localization/tsgv_data
- /new: 1024d 
- /org: activitynet (500d, C3D(4096d)-> PCA), charades (1024d, I3D), tacos (4096d, C3D), didemo (4096d, VGG)
