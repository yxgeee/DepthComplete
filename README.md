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
### Test
```
sh scripts/test.sh /your/dirname/to/model/ (eg. ./checkpoints/kitti/sparseconv_masked_maeloss)
```

## Experiments
|    Method    |  MAE  |  RMSE  |  iMAE  |  iRMSE  | 
| :----------- | :---: | :----: | :----: | :-----: |
| SparseConv(maeloss)   | 0.508195  | 1.730340   |  0.002170 | 0.006691 |
| SparseConv(log_maeloss)   | 0.530156  | 1.744890   |  0.002275 | 0.006768 |