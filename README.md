# Pytorch implementation of depth completion
(updating)

## Dependencies
- Python 2.7.*
- [PyTorch](http://pytorch.org/) (0.4.0)

## Support
- [Kitti](http://www.cvlibs.net/datasets/kitti/index.php) depth dataset
- SparseConv structure released by [Sparsity Invariant CNNs](http://arxiv.org/abs/1708.06500)

## TODO
- SparseConv baseline model training and evaluation, compared with different losses and optimization schedulers.
- Data augmentation rules.
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
## Train
```
sh scripts/train.sh
```

