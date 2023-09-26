## Introduction  
3DCrowd+ is presented as an enhanced iteration of the seminal 3DCrowdNet. This refined model adeptly modifies the baseline's coordattitude estimation network by amalgamating the attention mechanism with the model pruning algorithm.

<img width="1347" alt="截屏2023-09-26 22 09 23" src="https://github.com/bigmmgz/3D-Crowd-Reconstruction-from-a-Single-Image/assets/114939430/fab31523-7405-4d4d-b5aa-ca4f4558fbfb">



## Installation
Install Miniconda virtual environment. 
Install PyTorch =1.7.1 and Python = 3.7.3. 
  
  
## Preparing 
### In accordance with 3DCrowdNet
* Download the pre-trained 3DCrowdNet checkpoint from (https://drive.google.com/drive/folders/1YYQHbtxvdljqZNo8CIyFOmZ5yXuwtEhm?usp=sharing)
* Download `J_regressor_extra.npy` from [here](https://drive.google.com/file/d/1B9e65ahe6TRGv7xE45sScREAAznw9H4t/view?usp=sharing)

## Training
* Run 'python train.py --amp --continue --gpu 0-3 --cfg ../assets/yaml/3dpw_crowd.yml'

## Testing
* Run `python test.py --gpu 0`. You can change the input image with `--img_idx {img number}`.

## Evaluate
* Run 'python test.py --gpu 0-3 --cfg ../assets/yaml/3dpw.yml --exp_dir ../output/exp_08-07_23:23 --test_epoch 10'. You can replace the `--exp_dir` with your own experiments.


