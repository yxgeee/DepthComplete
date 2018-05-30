# Pytorch implementation of depth completion structures

## Dependencies
- Python 2.7.*
- [PyTorch](http://pytorch.org/) (0.4.0)

## Support
- [Kitti](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) depth complete dataset
- SparseConv structure released by [Sparsity Invariant CNNs](http://arxiv.org/abs/1708.06500)
- Sparse-to-dense structure (no RGB guided) released by [Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image](https://arxiv.org/pdf/1709.07492.pdf)

## Usage

### Download the code
```
git clone -b release http://path/to/this/repository/
```

### Prepare the data
create the data folder by
```
mkdir data
mkdir data/kitti
```
download [Kitti](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) dataset, and unzip all files into the same base directory (./data/kitti), the folder structure will look like this
```
|-- data
  |-- kitti
    |-- devkit
    |-- depth_selection
      |-- test_depth_completion_anonymous
      |-- val_selection_cropped
    |-- train
      |-- 2011_xx_xx_drive_xxxx_sync
        |-- proj_depth
          |-- groundtruth
          |-- velodyne_raw
      |-- ... (all drives of all days in the raw KITTI dataset)
    |-- val
      |-- (same as in train)
```

### Train
modify the training script, run by
```
sh scripts/train.sh
```
and scripts are referred to [experiments](#experiments).

### Generate dense depth images and test by official tools
compile the benchmark evaluation code
```
cd data/kitti/devkit/cpp
sh make.sh
```
validate on the selected valset (1000 images of size 1216x352, cropped and manually), run the testing script following by the folder path of checkpoints, the gpu device id, and the architecture name, eg.
```
sh scripts/test.sh ./checkpoints/kitti/sparseconv/masked_maeloss_adam 0 sparseconv
```
the evaluation logs and generated images will be save in `./checkpoints/kitti/sparseconv/masked_maeloss_adam/results`.

## <a name="experiments"></a>Experiments
The following results is evaluated on [Kitti](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) selected valset, and the training scripts are only for reference, maybe not necessarily optimal. **Download the models by clicking on the name of methods.**

|    Method                 | loss   |   MAE    |  RMSE    |  iMAE    |  iRMSE   |   Training Script     |
| :------------------------ | :----: | :------: | :------: | :------: | :------: | :----------- |
| [SparseConv](https://drive.google.com/open?id=1uC0MR9q4donBt_EDy66UBqK7H_H6olwD) | mae | 0.484260 | 1.777299 | 0.001947 | 0.006476 | python main.py --gpu-ids 0,1,2,3,4,5 -a sparseconv -b 64 --epochs 40 --step-size 20 --eval-step 1 --lr 0.001 --gamma 0.5 --criterion masked_maeloss --tag adam --optim adam |
| [SparsetoDense(d)](https://drive.google.com/open?id=1hgPwwkenRhHLP7WnMtCTZcqKyAMNvYbc) | mae | 0.425472 | 1.670506 | 0.001736 | 0.005809 | python main.py --gpu-ids 0,1 -a sparsetodense -b 32 --epochs 40 --step-size 8 --eval-step 1 --lr 0.01 --gamma 0.5 --criterion masked_maeloss --tag sgd --optim sgd |
