# ScreenshopColor

```
optional arguments:
  -h, --help            show this help message and exit
  --load pre_trained_weights_file
                        Path to weights file to be loaded. If specified, will
                        train model.
  --save new_weights_file
                        Path to saved weights file. If specified, will load
                        weights.
  --data-dir DATA_DIR   Path to data
  --cuda CUDA           If set to false, will not use GPU. defaults to False
  --epochs EPOCHS       Specify the number of epochs for training
  --batch BATCH         Batch size when training
  --lr LR               Learning rate
  --sample-seed SAMPLE_SEED
                        Seed for random sampling of dataset
```

```
python main.py --save myweights.pth --data-dir ./good_data --sample-seed 29
```


Using AlexNet to predict hsv values of images.

Run seperate_bad.py first to seperate samples with missing url or color

NOTE: Error when loading png url's. returns non-3 channel tensor. 
      Run product_dataset.py to download all images before training.