# How to implement MoLEF
We provide a skeleton code for potential authors of their own MLSV algorithm to add our MoLEF. Following the instructions in that file leads you to build your own algorithm and add to our MoLEF easily. You need to implement the following functions.

## Dataset 
To experiment with the custom dataset, you need to create the training and testing files in JSON format. These files should include the following information: the name of the video feature, duration, timestamps, sentences, words, and indices for the words.
```
[["v_yINX46xPRf0", # video feauture name
159.99, # duration
[0, 31.2], # timestamps
"The people are in the pool wearing goggles.", # sentence
["the", "people", "are", "in", "the", "pool", "wearing", "goggles"], # words
[0, 1, 2, 3, 4, 5, 6, 7], # id2pos,
]

```

## Constructors 
### CustomDataset 
The `CustomDataset` has 5 arguments, including feature path, word2vec, the number of frames, the number of words, and adjacency matrix. You may add your custom member variables to the class, then please make sure that they are correctly initialized in constructors. 

### CustomModel
The `CustomModel` has the subsequent structure, which loads `torch.nn.Module`. Following this structure, you are required to construct your own model to yield both outputs and loss, along with the Model class name. You can incorporate your custom model modules into the class, along with any necessary variables within the arguments.

```
class Model(nn.Module) :
  def __init__(self) :
      video_encoder = CustomVidEncoder()
      text_encoder = CustomTextEncoder()
  def forward(self, batch) :
      ....
      return outputs, loss
```

## Model Builder 
### build_model
If you create your own model, you have to encapsulate it to establish a running state. Within the MoLEF, you can find a `build.py` located within the runners folder. You should then integrate your model into the structure as outlined.

```
def build_model() : 
  from models.{model_name} import Model
    model = Model()
  
  return model 
```
### build_forward
To put the model in a running state, you need to add `cuda()` to the model inputs, and then create a dictionary. In the MoLEF, you can locate a `build_forward` function in `build.py` and subsequently pass the model inputs to your model.

```
def build_forward(batch, model) :
  vid_feats, word_feats, ... = batch
  model_inputs = {'feats': vid_feats.cuda(), 'words': word_feats.cuda(), ...}
  outputs, loss = self.model(**model_inputs)

  return outputs, loss
```

## Script 
We also provide training and evaluation scripts. The training script requires the `training` mode, and the evaluation script requires the `evaluation` mode. You can utilize different paths for video features and text features to experiment with your algorithm. Additionally, you have the flexibility to include hyperparameters as arguments, such as epochs, warmup updates, warmup initial learning rate, learning rate, weight decay, and more. 

### Training
```
python main.py --mode train --model model_name --word2vec-path  data/glove.840B.300d.bin \
--dataset Tacos --feature-path data/tacos/org --train-data data/tacos/train_data.json \
--val-data data/tacos/val_data.json --test-data data/tacos/test_data.json \
--max-num-epochs 20 --warmup-updates 300 --warmup-init-lr 1e-06 --lr 8e-4 \
--weight-decay 1e-7 --model-saved-path results/ --cfg code/configs/model_name.yml 
```
### Evaluation 
```
python main.py --mode evaluation --model model_name --word2vec-path  data/glove.840B.300d.bin \
--dataset Tacos --feature-path data/tacos/org  --train-data data/tacos/train_data.json \
--val-data data/tacos/val_data.json  --test-data data/tacos/test_data.json \
--model-load-path results/model_name --cfg code/configs/model_name.yml 
```
