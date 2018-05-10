## Prepare the data
unzip all downloaded zip files in the root path of kitti directory
```
mkdir data
ln -s /your/path/to/kitti/ data/kitti
```

## Train
```
python main.py --gpu-ids 0,1,2,3 -b 32 --save-root checkpoints/kitti
```