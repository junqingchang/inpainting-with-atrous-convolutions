# Image Inpainting using Atrous Convolutions
Making use of the Atrous Spatial Pyramid Pooling in DeeplabV3 to aid in image inpainting

## Directory Structure
```
data/
flowerschkpt/
flowerschkptretrain/
flowersplots/
flowersplotsretrain/
indoorchkpt/
indoorplots/

.gitignore
atrousinpainter.py
evaluate-flowers.py
evaluate-indoor.py
flowers.py
indoorscenerecognition.py
README.md
retrain-flowers.py
train-flowers.py
train-indoor.py
```

## Dataset
[Indoor Scene Recognition](http://web.mit.edu/torralba/www/indoor.html)
[Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

## Pretrained Models
Our pretrained models can be found in https://drive.google.com/drive/folders/1CUVoRSA5fsTQlA1FujKK8SXpfGAvZEoD?usp=sharing

## Training
To train new models

For Flowers102
```
$ python train-flowers.py
```

For Indoor Scene Recognition
```
$ python train-indoor.py
```

## Evaluation
To evaluate models
```
$ python evaluate-flowers.py
$ python evaluate-indoor.py
```