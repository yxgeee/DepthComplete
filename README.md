# Pytorch implementation of depth completion (updating)

## Dependencies
- Python 2.7.*
- [PyTorch](http://pytorch.org/) (0.4.0)

## Support
- [Kitti](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) depth complete dataset
- SparseConv structure released by [Sparsity Invariant CNNs](http://arxiv.org/abs/1708.06500)
- Sparse-to-dense structure released by [Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image](https://arxiv.org/pdf/1709.07492.pdf)

## TODO
- Data augmentation rules. (generate sparse data, etc.)
- SparseConv based residual structure, u-net structure, etc.
- Crf post-process or end-to-end training
- RGB guided

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
sh scripts/test.sh ./checkpoints/kitti/sparseconv_masked_maeloss 0 sparseconv
```

## Experiments
### Evaluate on [Kitti](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) selected val set
|    Method                 |    MAE   |  RMSE    |  iMAE    |  iRMSE   |   Script     |
| :------------------------ | :------: | :------: | :------: | :------: | :----------- |
| SparseConv(maeloss)       | 0.462935 | 1.731169 | 0.001959 | 0.006377 | python main.py --gpu-ids 0,1,2 -a sparseconv -b 32 --epochs 20 --step-size 0 --eval-step 1 --lr 0.001 --criterion masked_maeloss --optim adam |
| SparseConv(log_maeloss)   | 0.479404 | 1.738473 | 0.002044 | 0.006431 | python main.py --gpu-ids 0,1,2 -a sparseconv -b 32 --epochs 20 --step-size 0 --eval-step 1 --lr 0.001 --criterion masked_log_maeloss --optim adam |