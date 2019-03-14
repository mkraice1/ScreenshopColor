# ScreenshopColor

Using models to predict hsv values of images.

NOTE: Error when loading png url's. returns non-3 channel tensor. 
      Run product_dataset.py to download all images before training.

- Run seperate_bad.py first to seperate samples with missing url or color
- Run display_results after training a model to see images and the color 
  strings they are tagged with


```
optional arguments for main:
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
Train a model example:
 python main.py --save res34weights_mse_1.pth --epochs 5 --model resnet34 
--loss mse

Test a model example:
python .\main.py --load res34weights_mse_1.pth
```


```
optional arguments for display_results:
  -h, --help            show this help message and exit
  --load WEIGHTS_FILE   Path to weights file to be loaded. If specified, will
                        train model.
  --data-dir DATA_DIR   Path to data (default: ./good_data).
  --model MODEL_TYPE    Type of model (default: alexnet): alexnet, resnet18,
                        resnet34
  --num-samples NUM_SAMPLES
                        Number of samples to test against
```

```
Example:

python .\display_results.py --load res34weights_mse_1.pth --model resnet34 
  --num-samples 18
```