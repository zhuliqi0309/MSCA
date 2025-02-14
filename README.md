# Introduction
Implementation of the paper [Enhancing Image-Text Matching through Multi-Level Semantic Consistency Alignment].
The contributing journals are The Visual Computer.

* We have released most of our codebase. The code is now usable for training our model using GRU and Glove as the text encoder.

* TODO: release the code and config to train our model wih GloVe embedding and Bi-GRU text encoder on both Flick30K and COCO dataset.

## Prerequisites
### Environment
The experiments in this project were done with the following key dependencies:
* python    3.7.16
* pytorch   1.8.0
* numpy    1.21.6
* tensorboard    2.11.2
* nltk   3.8.1

### Data
Download the dataset files. We use the image feature created by SCAN. The vocabulary required by GloVe has been placed in the 'vocab' folder of the project (for Flickr30K and MSCOCO).

You can download the dataset through Baidu network disk. Download links are [Flickr30K and MSCOCO](https://pan.baidu.com/s/1JDlsr7s9HKMOSCUGCUh-_Q?pwd=ZLLQ ,the extration code is : ZLLQ).


## Pretrained checkpoints
If you don't want to train from scratch, you can download the pretrained MSCA model from [here](https://pan.baidu.com/s/1gcIa4NEv9iyAXsCknhd7Ww?pwd=ZLQQ 提取码: ZLQQ)(for Flickr30K model without using GloVe) and [here]( https://pan.baidu.com/s/1Vq8vRlgXghiw6TFoyweqlw?pwd=ZLLQ 提取码: ZLLQ)(for Flickr30K model  using GloVe). For example, the following is the result of a Glove run on the Flickr 30K dataset.
```
rsum: 530.2
Average i2t Recall: 92.3
Image to text: 83.2 95.8 98.0 1.0 2.1
Average t2i Recall: 84.4
Text to image: 70.2 89.5 93.5 1.0 5.2

```

## Training
All configuration parameters can be found in the 'train.py' file. To run training, find the config file corresponding to the settings you want to run, make adjustment for your environment, hyperparams, etc., and launch training with
```
# python train.py --data_path "$DATA_PATH" --data_name f30k_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/log --logg_path runs/runX/logs --model_name "$MODEL_PATH" 
```

## Test
Test on Flickr30K
```
# python test.py
```
To do cross-validation on MSCOCO, pass fold5=True with a model trained using --data_name coco_precomp.
```
#python testall.py
```