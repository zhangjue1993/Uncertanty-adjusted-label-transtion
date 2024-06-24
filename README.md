# Weakly Supervised Solar Panel Mapping via Uncertainty Adjusted Label Transition in Aerial Images
Source code for "**Weakly Supervised Solar Panel Mapping via Uncertainty Adjusted Label Transition in Aerial Images
**", accepted in IEEE TIP. The paper's PDF can be found [Here](https://ieeexplore.ieee.org/abstract/document/10351041/).

### data sets
GoogleEarth Static Map API

# Parameters
Please refer to ```act_config.json```

## The generation of Pseudo labels 
Use torchCAM to implement grancam. The code can be found [Here](https://github.com/frgfm/torch-cam).

## Training
UALT consists of three part Unertainty Estimation Network, Label Transition Network and Target Mapping Network, which are trained in sequence. 

Setting the training data to the proper root as follows:


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

### Label Transition Network

```
set ```self-pace``` to ```False``` in ```act_config.json```. 

Run ```python train.py --config_path act_config.json```.

### 3nd training stage

Set ```self-pace``` to ```True``` in ```act_config.json```. Set the label update dir in ```act_config.json```. Run  ```python train.py --config_path act_config.json```.

## Testing
```python predict.py --config_path act_config.json```

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
