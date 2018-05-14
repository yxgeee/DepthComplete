# Pytorch implementation of depth completion (updating)

## Dependencies
- Python 2.7.*
- [PyTorch](http://pytorch.org/) (0.4.0)

## Support
- [Kitti](http://www.cvlibs.net/datasets/kitti/index.php) depth dataset
- SparseConv structure released by [Sparsity Invariant CNNs](http://arxiv.org/abs/1708.06500)
- Sparse-to-dense structure released by [Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image](https://arxiv.org/pdf/1709.07492.pdf)

## TODO
- SparseConv baseline model training and evaluation, compared with different losses and optimization schedulers.
- Data augmentation rules. (generate sparse data, crop size)
- SparseConv based residual structure, u-net struture, etc.
- Crf

## How to use

### Download dataset (eg. Kitti)
```
wget http://kitti.is.tue.mpg.de/kitti/data_depth_velodyne.zip
wget http://kitti.is.tue.mpg.de/kitti/data_depth_annotated.zip
wget http://kitti.is.tue.mpg.de/kitti/data_depth_selection.zip
wget http://www.cvlibs.net/downloads/depth_devkit.zip
```
### Prepare the data folder
unzip all downloaded zip files in the root path of kitti directory
```
mkdir data
ln -s /your/path/to/kitti/ data/kitti
```
### Train
```
sh scripts/train.sh
```

## Experiments
#### SparseConv with masked-maeloss
```
mean mae: 0.508195 
mean rmse: 1.730340 
mean inverse mae: 0.002170 
mean inverse rmse: 0.006691 
mean log mae: 0.027238 
mean log rmse: 0.079132 
mean scale invariant log: 0.078568 
mean abs relative: 0.030776 
mean squared relative: 0.016117 
```
|    method    |  MAE  |  RMSE  |
| :----------- | :---: | :----: |
| SparseConv   | 0.51  | 1.73   |