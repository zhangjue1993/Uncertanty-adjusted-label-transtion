# Weakly Supervised Solar Panel Mapping via Uncertainty Adjusted Label Transition in Aerial Images
Source code for "**Weakly Supervised Solar Panel Mapping via Uncertainty Adjusted Label Transition in Aerial Images
**", accepted in IEEE TIP. The paper's PDF can be found [Here](https://ieeexplore.ieee.org/abstract/document/10351041/).


## Parameters
Please refer to ```act_config.json``` in each folder

## The generation of Pseudo labels 
Use torchCAM to implement grandcam. The code can be found [Here](https://github.com/frgfm/torch-cam).

## Training
UALT consists of three parts: Uncertainty Estimation Network, Label Transition Network, and Target Mapping Network, which are trained in sequence. 

### Unertainty Estimation Network 
run ```python train.py --config_path act_config.json``` in folder UEN. 
For UEN, set the training directory as follows:

```
-- data -- train -- fore

                 -- back
                 
                 -- Pseudo labels
                
         -- test -- cls -- fore
         
                        -- back
                        
                 -- seg -- img
                 
                        -- gt
```
Inference: Generate **Bayes_optimal_label** and **Train_sigma_dir** for LTN training with  ```python predict0618.py --config_path act_config.json```

### Label Transition Network

Run ```python train.py --config_path act_config.json```.

For LTN, set the training directory as follows:

```
-- data -- train -- fore

                 -- back
                 
                 -- Pseudo labels

                 -- Bayes_optimal_label

                 -- Train_sigma_dir
```
Inference: Generate IDTM estimation for each image in the training dataset with  ```python predict.py --config_path act_config.json```

### Target Mapping Network

Will be updated soon.....

## Citation
We really hope this repo can contribute to the community, and if you find this work useful, please use the following citation:
```
@ARTICLE{10351041,
  author={Zhang, Jue and Jia, Xiuping and Zhou, Jun and Zhang, Junpeng and Hu, Jiankun},
  journal={IEEE Transactions on Image Processing}, 
  title={Weakly Supervised Solar Panel Mapping via Uncertainty Adjusted Label Transition in Aerial Images}, 
  year={2024},
  volume={33},
  number={},
  pages={881-896},
  doi={10.1109/TIP.2023.3336170}}
