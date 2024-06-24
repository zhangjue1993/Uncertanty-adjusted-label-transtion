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
Generate **Bayes_optimal_label** and **Train_sigma_dir** for LTN training

## Testing

```python predict0618.py --config_path act_config.json```

### Label Transition Network

Run ```python train.py --config_path act_config.json```.

For LTN, set the training directory as follows:

```
-- data -- train -- fore

                 -- back
                 
                 -- Pseudo labels

                 -- Bayes_optimal_label

                 -- Train_sigma_dir




**Noting** that the results in our paper do not adopt any post-process including CRF.

The evaluation code can be found in [here](https://github.com/zhangjue1993/torch--SP-RAN-Self-paced-Residual-Aggregated-Network/blob/main/Evaluate.py).


## Contact me
If you have any questions, pleas feel free to contact me: jue.zhang@adfa.edu.au.


## Citation
We really hope this repo can contribute the conmunity, and if you find this work useful, please use the following citation:
```
@ARTICLE{9585690,
  author={Zhang, Jue and Jia, Xiuping and Hu, Jiankun},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SP-RAN: Self-Paced Residual Aggregated Network for Solar Panel Mapping in Weakly Labeled Aerial Images}, 
  year={2022},
  volume={60},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2021.3123268}}
