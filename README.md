## Introduction  
3DCrowd+ is presented as an enhanced iteration of the seminal 3DCrowdNet. This refined model adeptly modifies the baseline's attitude estimation network by amalgamating the attention mechanism with the model pruning algorithm.

## Installation
Install Miniconda virtual environment. 
Install PyTorch =1.7.1 and Python = 3.7.3. 
  
  
## Preparing 
### In accordance with 3DCrowdNet
* Download the pre-trained 3DCrowdNet checkpoint from (https://drive.google.com/drive/folders/1YYQHbtxvdljqZNo8CIyFOmZ5yXuwtEhm?usp=sharing)
* Get SMPL layers and VPoser according to [this](./assets/directory.md#pytorch-smpl-layer-and-vposer).
* Download `J_regressor_extra.npy` from [here](https://drive.google.com/file/d/1B9e65ahe6TRGv7xE45sScREAAznw9H4t/view?usp=sharing)

## Training
* Run 'python train.py --amp --continue --gpu 0-3 --cfg ../assets/yaml/3dpw_crowd.yml'

## Testing
* Run `python test.py --gpu 0`. You can change the input image with `--img_idx {img number}`.

## Evaluate
* Run 'python test.py --gpu 0-3 --cfg ../assets/yaml/3dpw.yml --exp_dir ../output/exp_04-06_23:43 --test_epoch 10'. You can replace the `--exp_dir` with your own experiments.


