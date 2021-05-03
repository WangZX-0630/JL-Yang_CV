## Improved Road Connectivity by Joint Learning of Orientation and Segmentation ##
#### In CVPR 2019 [[pdf]](https://anilbatra2185.github.io/papers/RoadConnectivityCVPR2019.pdf) [[supplementary]](https://anilbatra2185.github.io/papers/RoadConnectivity_CVPR_Supplementary.pdf)



## Requirements
* [PyTorch](https://pytorch.org/) (version = 1.5.0)
* numba
* json
* skimage
* numpy
* tqdm

## Data Preparation


*Download DeepGlobe Road dataset in the following tree structure.*

```
data/deepglobe
|   train.txt
|   val.txt
|
└───train
│   └───gt
│   └───images
└───val
│   └───gt
│   └───images

```


## Training

Train Multi-Task learning framework to predict road segmentation and road orientation.

__Training MTL Help__
```
file:
        train_mtl.py   training file for road connectivity phase 1

                    
usage: train_mtl.py --config CONFIG
                    --model_name {LinkNet34MTL,StackHourglassNetMTL}
                    --dataset {deepglobe,spacenet}
                    --exp EXP
                    [--resume RESUME]
                    [--model_kwargs MODEL_KWARGS]
                    [--multi_scale_pred MULTI_SCALE_PRED]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       config file path
  --model_name 			{LinkNet34MTL,StackHourglassNetMTL}
                        Name of Model = ['StackHourglassNetMTL',
                        'LinkNet34MTL']
  --exp EXP             Experiment Name/Directory
  --resume RESUME       path to latest checkpoint (default: None)
  --dataset 			{deepglobe,spacenet}
                        select dataset name from ['deepglobe', 'spacenet'].
                        (default: Spacenet)
  --model_kwargs 		MODEL_KWARGS
                        parameters for the model
  --multi_scale_pred 	MULTI_SCALE_PRED
                        perform multi-scale prediction (default: True)
                        
```

__Sample Usage__

* Training with StackModule
```
CUDA_VISIBLE_DEVICES=0,1 python train_mtl.py --config config.json --dataset deepglobe --model_name "StackHourglassNetMTL" --exp dg_stak_mtl
```
* Training with LinkNet34
```
CUDA_VISIBLE_DEVICES=0,1 python train_mtl.py --config config.json --dataset deepglobe --model_name "LinkNet34MTL" --exp dg_L34_mtl --multi_scale_pred false

```

__Testing MTL Help__

file:    test.py   similar usage with training but remember to add --resume (path of model)


## Evaluate APLS

* Use Java implementation to compute APLS provided by Spacenet Challenge. 
```
java -jar visualizerDG.jar -params ./data/paramsDG.txt
```


## Connectivity Refinement

* Training with Linear Artifacts/Corruption (using LinkNet34 Architecture)
```
file:   train_refine_pre.py    training file for road connectivity phase 2
        notice: 1. modify class DeepGlobeDatasetCorrupt in road_dataset_train.py to change pretrain or finetune
                2. modify line26 in road_dataset_train.py to change the data to use in finetune stage
```

```
usage(pretrain): CUDA_VISIBLE_DEVICES=0,1 python train_refine_pre.py --config config.json --dataset deepglobe --model_name "LinkNet34" --exp deepglobe_L34_pre_train_with_corruption --multi_scale_pred false

usage(finetune): CUDA_VISIBLE_DEVICES=0,1 python train_refine_pre.py --config config.json --dataset deepglobe --model_name "LinkNet34" --exp deepglobe_L34_fine_tune_with_corruption --multi_scale_pred false --resume /root/test/deepglobe_L34_pre_train_with_corruption/model_best.pth.tar

```

* Test

file:    test_refine_pre.py



## Notice
Remember to check the path and parameters in config.json everytime you do training or testing!!!



## Citation
If you find our work useful in your research, please cite:

    @InProceedings{Batra_2019_CVPR,
		author = {Batra, Anil and Singh, Suriya and Pang, Guan and Basu, Saikat and Jawahar, C.V. and Paluri, Manohar},
		title = {Improved Road Connectivity by Joint Learning of Orientation and Segmentation},
		booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
		month = {June},
		year = {2019}
	}

