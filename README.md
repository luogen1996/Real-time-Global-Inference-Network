# Global Inference Network- pytorch
This is the  implementation of the paper: A Real-time Global Inference Network for One-stage
Referring Expression Comprehension.

Our code is based on  [ZSGNet](https://github.com/TheShadow29/zsgnet-pytorch/tree/master/code ). We further add two modules, i.e. the  Adaptive Feature Selection and the Global Attentive ReAsoNing unit with an attention loss. Besides, we release all pretrained models and  datasets  used in our paper .

Note that the preparations of this code are following the setting of  [ZSGNet](https://github.com/TheShadow29/zsgnet-pytorch/tree/master/code). If you have any problems, please contact with [us](luogen@stu.xmu.edu.cn).

## Training
Basic usage is `python code/main_dist.py "experiment_name" --arg1=val1 --arg2=val2` and the arg1, arg2 can be found in `configs/cfg.yaml`. This trains using the DataParallel mode.

For distributed learning use `python -m torch.distributed.launch --nproc_per_node=$ngpus code/main_dist.py` instead. This trains using the DistributedDataParallel mode. (Also see [caveat in using distributed training](#caveats-in-distributeddataparallel) below)

An example to train on ReferIt dataset (note you must have prepared referit dataset) would be:

```
python code/main_dist.py "referit_try" --ds_to_use='refclef' --bs=16 --nw=4
```

Similarly for distributed learning (need to set npgus as the number of gpus)
```
python -m torch.distributed.launch --nproc_per_node=$npgus code/main_dist.py "referit_try" --ds_to_use='refclef' --bs=16 --nw=4
```

## Evaluation
There are two ways to evaluate. 

1. For validation, it is already computed in the training loop. If you just want to evaluate on validation or testing on a model trained previously ($exp_name) you can do:
```
python code/main_dist.py $exp_name --ds_to_use='refclef' --resume=True --only_val=True --only_test=True
```
or you can use a different experiment name as well and pass `--resume_path` argument like:
```
python code/main_dist.py $exp_name --ds_to_use='refclef' --resume=True --resume_path='./tmp/models/referit_try.pth' 
```
After this, the logs would be available inside `tmp/txt_logs/$exp_name.txt`

2. If you have some other model, you can output the predictions in the following structure into a pickle file say `predictions.pkl`:
```
[
    {'id': annotation_id,
 	'pred_boxes': [x1,y1,x2,y2]},
    .
    .
    .
]
```

Then you can evaluate using `code/eval_script.py` using:
```
python code/eval_script.py predictions_file gt_file
```
For referit it would be
```
python code/eval_script.py ./tmp/predictions/$exp_name/val_preds_$exp_name.pkl ./data/referit/csv_dir/val.csv
```

## Datasets 

|          Dataset          |                             Link                             |
| :-----------------------: | :----------------------------------------------------------: |
|         Flickr30k         | [One Drive](https://1drv.ms/u/s!AmrFUyZ_lDVGfbdSDe1auqCob_g?e=nR9TwQ) |
|          Referit          | [One Drive](https://1drv.ms/u/s!AmrFUyZ_lDVGfEjmz6IXI1q1rRc?e=jIi8kH) |
|      Flickr-Split-0       | [One Drive](https://1drv.ms/u/s!AmrFUyZ_lDVGe-XCz_Gh36JcxL4?e=NNjVkh) |
|      Flickr-Split-1       | [One Drive](https://1drv.ms/u/s!AmrFUyZ_lDVGegJXY6YO8rowCMI?e=F5Oeu1) |
|     VG-2B,2UB,3B,3UB      | [One Drive](https://1drv.ms/u/s!AmrFUyZ_lDVGfirzOmAE0CA00uo?e=1v2I6F) |
| RefCOCO,RefCOCO+,RefCOCOg |                         coming soon!                         |



## Pre-trained Models

we  tried  to repeat the results of ZSGNet. But unfortunately, the results are a bit  different from the paper, especially in referit, where the results are slightly better in our experiences.

| Model                             |  Dataset  |     val     |    test     |                             link                             |
| --------------------------------- | :-------: | :---------: | :---------: | :----------------------------------------------------------: |
| ZSGNet                            | flickr30K |    63.15    |    63.43    | [One Drive](https://1drv.ms/u/s!AmrFUyZ_lDVGf68FpDXLIlzvlzY?e=jiQ8CH) |
| GIN(10 epochs)                    | flickr30K |    64.06    |    64.77    | [One Drive](https://1drv.ms/u/s!AmrFUyZ_lDVGgQL6t2Cp4UvmzPiu?e=PIltdE) |
| GIN(10 epochs+ resized_10_epochs) | flickr30K |    66.54    |    68.14    |                                                              |
| ZSGNet                            |  referit  |    65.99    |    62.73    | [One Drive](https://1drv.ms/u/s!AmrFUyZ_lDVGgQCAeaYj4-Kt_OXX?e=V2bJJt) |
| GIN(10 epochs)                    |  referit  |    68.40    |    65.15    | [One Drive](https://1drv.ms/u/s!AmrFUyZ_lDVGgQErMsd5oJx-S9x-?e=dnjqMK) |
| GIN(10 epochs+ resized_10_epochs) |  referit  | coming soon | coming soon |                                                              |


