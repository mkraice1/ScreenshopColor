# ScreenshopColor

```
optional arguments:
  -h, --help            show this help message and exit
  --load PRE_TRAINED_WEIGHTS_FILE
                        Path to weights file to be loaded. If specified, will
                        train model.
  --save NEW_WEIGHTS_FILE
                        Path to saved weights file. If specified, will load
                        weights.
  --data-dir DATA_DIR   Path to data (default: ./good_data).
  --model MODEL_TYPE    Type of model (default: alexnet): alexnet, resnet18,
                        resnet50
  --loss LOSS           Loss function to use (default: bce): bce, mse
  --cuda CUDA           If set to true, will use GPU (default: False).
  --epochs EPOCHS       Specify the number of epochs for training (default:
                        5).
  --batch BATCH         Batch size when training (default: 4).
  --lr LR               Learning rate (default: .001).
  --sample-seed SAMPLE_SEED
                        Seed for random sampling of dataset (default: 42).
```

```
python main.py --save myweights.pth --data-dir ./good_data --sample-seed 29
```


Using AlexNet to predict hsv values of images.

Run seperate_bad.py first to seperate samples with missing url or color

NOTE: Error when loading png url's. returns non-3 channel tensor. 
      Run product_dataset.py to download all images before training.