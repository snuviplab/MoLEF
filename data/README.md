# Prepare data
directory structure

```
├── data 
    ├── activitynet
    │    ├── new 
    │    ├── org 
    │    ├── train_data.json
    │    ├── valid_data.json
    │    └── test_data.json
    ├── charades
    │    ├── new 
    │    ├── org
    │    └── ...
    ├── didemo
    │    ├── new
    │    ├── org
    │    └── ...
    └── tacos
        ├── new
        ├── org
        └── ...
```

- Download glove 300d file from [link](https://drive.google.com/file/d/1XOlwnO2lMeqio8A6pxHzPs3La0eD6sWk/view?usp=sharing) and should be placed in `data/`. 
- /new: all (1024d, I3D) 
- /org: activitynet (500d, C3D(4096d)-> PCA), charades (1024d, I3D), tacos (4096d, C3D), didemo (4096d, VGG), youcook2(512d, MIL-NCE), msrvtt(4096d, C3D), tvr (3072d, concat)
- Dataset used in this framework can be downloaded from [link](https://drive.google.com/file/d/1XOlwnO2lMeqio8A6pxHzPs3La0eD6sWk/view?usp=sharing) 
- {train, valid, test} json format: [video, duration, moment, token, sentence, wordidx, dependency graph]
- For dependency graph extraction, see the details in [link](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html)
