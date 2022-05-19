# [ICCV 2021] OMNet: Learning Overlapping Mask for Partial-to-Partial Point Cloud Registration

This is the Pytorch implementation of our ICCV2021 paper [OMNet](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_OMNet_Learning_Overlapping_Mask_for_Partial-to-Partial_Point_Cloud_Registration_ICCV_2021_paper.pdf). For our MegEngine implementation, please refer to [this repo](https://github.com/megvii-research/OMNet).

Our presentation video: [[Youtube](https://www.youtube.com/watch?v=u2lTKsom8oU)][[Bilibili](https://www.bilibili.com/video/BV1Ef4y1J7XP/)].

## Our Poster

![image](./images/OMNet_poster.png)

## Dependencies

* Pytorch>=1.5.0
* Other requirements please refer to`requirements.txt`.

## Data Preparation

### OS data

We refer the original data from PointNet as OS data, where point clouds are only sampled once from corresponding CAD models. We offer two ways to use OS data, (1) you can download this data from its original link [original_OS_data.zip](http://modelnet.cs.princeton.edu/). (2) you can also download the data that has been preprocessed by us from link [our_OS_data.zip](https://drive.google.com/file/d/1rXnbXwD72tkeu8x6wboMP0X7iL9LiBPq/view?usp=sharing).

### TS data

Since OS data incurs over-fitting issue, we propose our TS data, where point clouds are randomly sampled twice from CAD models. You need to download our preprocessed ModelNet40 dataset first, where 8 axisymmetrical categories are removed and all CAD models have 40 randomly sampled point clouds. The download link is [TS_data.zip](https://drive.google.com/file/d/1DPBBI3Ulvp2Mx7SAZaBEyvADJzBvErFF/view?usp=sharing). All 40 point clouds of a CAD model are stacked to form a (40, 2048, 3) numpy array, you can easily obtain this data by using following code:

```
import numpy as np
points = np.load("path_of_npy_file")
print(points.shape, type(points))  # (40, 2048, 3), <class 'numpy.ndarray'>
```

Then, you need to put the data into `./dataset/data`, and the contents of directories are as follows:

```
./dataset/data/
├── modelnet40_half1_rm_rotate.txt
├── modelnet40_half2_rm_rotate.txt
├── modelnet_os
│   ├── modelnet_os_test.pickle
│   ├── modelnet_os_train.pickle
│   ├── modelnet_os_val.pickle
│   ├── test [1146 entries exceeds filelimit, not opening dir]
│   ├── train [4194 entries exceeds filelimit, not opening dir]
│   └── val [1002 entries exceeds filelimit, not opening dir]
└── modelnet_ts
    ├── modelnet_ts_test.pickle
    ├── modelnet_ts_train.pickle
    ├── modelnet_ts_val.pickle
    ├── shape_names.txt
    ├── test [1146 entries exceeds filelimit, not opening dir]
    ├── train [4196 entries exceeds filelimit, not opening dir]
    └── val [1002 entries exceeds filelimit, not opening dir]
```

## Training and Evaluation

### Begin training

For ModelNet40 dataset, you can just run:

```
python3 train.py --model_dir=./experiments/experiment_omnet/
```

For other dataset, you need to add your own dataset code in `./dataset/data_loader.py`. Training with a lower batch size, such as 16, may obtain worse performance than training with a larger batch size, e.g., 64.

### Begin testing

You need to download the pretrained checkpoint and run:

```
python3 evaluate.py --model_dir=./experiments/experiment_omnet --restore_file=./experiments/experiment_omnet/val_model_best.pth
```

The following table shows our performance on ModelNet40, where `val` and `test` indicate  `Unseen Shapes` and `Unseen Categories` respectively. `PRNet` and `RPMNet` indicate the partial manners used in [PRNet](https://arxiv.org/pdf/1910.12240.pdf) and [RPMNet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yew_RPM-Net_Robust_Point_Matching_Using_Learned_Features_CVPR_2020_paper.pdf) respectively.

|     dataset     |     | RMSE(R) | MAE(R) | RMSE(t) | MAE(t) | Error(R) | Error(t) |                                           checkpoint                                           |
| :-------------: | :--: | :-----: | :----: | :-----: | :----: | :------: | :------: | :---------------------------------------------------------------------------------------------: |
| OS_PRNet_clean | val |  0.912  | 0.339 | 0.0078 | 0.0049 |  0.639  |  0.0099  | [Google Drive](https://drive.google.com/file/d/1i6nsSPFriGYxD1rDGTpbtTBYmdQcT8St/view?usp=sharing) |
|                 | test |  2.247  | 0.652 | 0.0177 | 0.0077 |  1.241  |  0.0154  | [Google Drive](https://drive.google.com/file/d/1LTR4rCT4eQ6JXXOekeUjhXQpwhlh9NEY/view?usp=sharing) |
| TS_PRNet_clean | val |  1.032  | 0.506 | 0.0085 | 0.0057 |  0.984  |  0.0113  | [Google Drive](https://drive.google.com/file/d/1AdutxYe7FS88uoLMf7V6Mo9Tlb9hSDlF/view?usp=sharing) |
|                 | test |  2.372  | 0.974 | 0.0146 | 0.0077 |  1.892  |  0.0152  | [Google Drive](https://drive.google.com/file/d/1A-6xTPGPAbmnwbnt81NjhN6Mw9VMHmwN/view?usp=sharing) |
| OS_PRNet_noise | val |  1.029  | 0.573 | 0.0089 | 0.0061 |  1.077  |  0.0123  | [Google Drive](https://drive.google.com/file/d/1JbBlBW08PQrucbdpp-G-VlWdjlik6tO7/view?usp=sharing) |
|                 | test |  2.318  | 0.957 | 0.0155 | 0.0078 |  1.809  |  0.0156  | [Google Drive](https://drive.google.com/file/d/154xYpstuQJ0eDk3rqbDShg5P5b17xlry/view?usp=sharing) |
| TS_PRNet_noise | val |  1.314  | 0.771 | 0.0102 | 0.0074 |  1.490  |  0.0148  | [Google Drive](https://drive.google.com/file/d/1ZzetsjHC4POh8Irr1RfSl8boPJvCQFMx/view?usp=sharing) |
|                 | test |  2.443  | 1.189 | 0.0181 | 0.0097 |  2.311  |  0.0193  | [Google Drive](https://drive.google.com/file/d/1eHi9pzAmL3jrYGmv6X9xy-8U7hAw9OdI/view?usp=sharing) |
| OS_RPMNet_clean | val |  0.771  | 0.277 | 0.0154 | 0.0056 |  0.561  |  0.0122  | [Google Drive](https://drive.google.com/file/d/1_wGJTxaezFvb4xqABmFfrIR03Wq2c80U/view?usp=sharing) |
|                 | test |  3.719  | 1.314 | 0.0392 | 0.0151 |  2.659  |  0.0321  | [Google Drive](https://drive.google.com/file/d/1IQ0DZ_OmaZPErPqm4DJ4NBWfnt1d1YG5/view?usp=sharing) |
| TS_RPMNet_clean | val |  1.401  | 0.544 | 0.0241 | 0.0095 |  1.128  |  0.0202  | [Google Drive](https://drive.google.com/file/d/1IlUSzGoAXHzon5ZrwLPNBsTuICphhrAO/view?usp=sharing) |
|                 | test |  4.016  | 1.622 | 0.0419 | 0.0184 |  3.205  |  0.0394  | [Google Drive](https://drive.google.com/file/d/1NJZcfHoXlCFTMVz01ZACiiTMNUEW1QQC/view?usp=sharing) |
| OS_RPMNet_noise | val |  0.998  | 0.555 | 0.0172 | 0.0078 |  1.079  |  0.0167  | [Google Drive](https://drive.google.com/file/d/1LvhPwrtUs-A2AZWO1YgrvhgcxdZXTen-/view?usp=sharing) |
|                 | test |  3.572  | 1.570 | 0.0391 | 0.0172 |  3.073  |  0.0359  | [Google Drive](https://drive.google.com/file/d/1xnHcKikXs8D9UuGchwo3YK21vG86zRtp/view?usp=sharing) |
| TS_RPMNet_noise | val |  1.522  | 0.817 | 0.0189 | 0.0098 |  1.622  |  0.0208  |                                                -                                                |
|                 | test |  4.356  | 1.924 | 0.0486 | 0.0223 |  3.834  |  0.0476  |                                                -                                                |

## Citation

```
@InProceedings{Xu_2021_ICCV,
    author={Xu, Hao and Liu, Shuaicheng and Wang, Guangfu and Liu, Guanghui and Zeng, Bing},
    title={OMNet: Learning Overlapping Mask for Partial-to-Partial Point Cloud Registration},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month={October},
    year={2021},
    pages={3132-3141}
}
```

## Acknowledgments

In this project we use (parts of) the official implementations of the following works:

* [RPMNet](https://github.com/yewzijian/RPMNet) (ModelNet40 preprocessing and evaluation)
* [PRNet](https://github.com/WangYueFt/prnet) (ModelNet40 preprocessing)

We thank the respective authors for open sourcing their methods.
