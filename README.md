# ScreenshopColor

optional arguments:
  -h, --help            show this help message and exit
  --load PRE_TRAINED_WEIGHTS_FILE
                        Path to weights file to be loaded. If specified, will
                        train model.
  --save NEW_WEIGHTS_FILE
                        Path to saved weights file. If specified, will load
                        weights.
  --data-dir DATA_DIR   Path to data
  --cuda CUDA           If set to false, will not use GPU. defaults to False
  --epochs EPOCHS       Specify the number of epochs for training
  --batch BATCH         Batch size when training
  --lr LR               Learning rate
  --sample-seed SAMPLE_SEED
                        Seed for random sampling of dataset



Using AlexNet to predict hsv values of images.
