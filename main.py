from product_dataset import ProductDataset, color_to_hsv_fn, model_out_to_color_fn
from torchvision import transforms
from torchvision.models import alexnet, resnet18, resnet34
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import BCEWithLogitsLoss, MSELoss, Linear
from torch.optim import Adam

from torch import save, load

import matplotlib
import matplotlib.pyplot
import numpy as np
import argparse
import math
import time
import copy
import random

INPUT_SIZE  = 224

# TODO
# add a quick test option
# add function doc strings
# add arguments for diff data split
# Save model based off num correct?
# Better management of passing args around
# Better way to get length of dataset



def main():

    # Comman line interface
    parser = argparse.ArgumentParser(description='Regression color img to hsv')
    parser.add_argument('--load', dest='pre_trained_weights_file',
                        help='Path to weights file to be loaded. If specified, will train model.')
    parser.add_argument('--save', dest='new_weights_file',
                        help='Path to saved weights file. If specified, will load weights.')
    parser.add_argument('--data-dir', dest='data_dir', default="./good_data",
                        help='Path to data (default: ./good_data).')
    parser.add_argument('--model', dest='model_type', default="alexnet",
                        help='Type of model (default: alexnet): alexnet, resnet18, resnet50')
    parser.add_argument('--loss', dest='loss', default="bce",
                        help='Loss function to use (default: bce): bce, mse')
    parser.add_argument('--cuda', dest='cuda', default="False",
                        help='If set to true, will use GPU (default: False).')
    parser.add_argument('--epochs', dest='epochs', default=5, type=int,
                        help='Specify the number of epochs for training (default: 5).')
    parser.add_argument('--batch', dest='batch', default=4, type=int,
                        help='Batch size when training (default: 4).')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float,
                        help='Learning rate (default: .001).')
    parser.add_argument('--sample-seed', dest='sample_seed', default=42, type=int,
                        help='Seed for random sampling of dataset (default: 42).')
    args = parser.parse_args()


    # Using cuda?
    if args.cuda == "True":
        do_cuda = True
        print("Using GPU")
    else:
        do_cuda = False


    # Select loss function
    if args.loss.lower() == "bce":
        loss_fn = BCEWithLogitsLoss()
    elif args.loss.lower() == "mse":
        loss_fn = MSELoss()
    else:
        print( "Invalid loss selected" )
        exit()


    # Load data and models
    num_outputs = 3
    data_loaders = prep_data( args.batch, args.sample_seed, args.data_dir )
    if args.model_type == "alexnet":
        model = alexnet( pretrained=False )
        model.classifier[6] = Linear( 4096, num_outputs )

    elif args.model_type == "resnet18":
        model = resnet18( pretrained=False )
        model.fc = Linear( 512, num_outputs )

    elif args.model_type == "resnet34":
        model = resnet34( pretrained=False )
        model.fc = Linear( 512, num_outputs )
        
    else:
        print("No model selected")


    # Train or test a model 
    if args.new_weights_file and args.pre_trained_weights_file:
        print( "Only specify a save or load file. Not both." )

    # Train the model
    elif args.new_weights_file:
        print( "Training model..." )
        train( model, args.epochs, args.new_weights_file, args.lr, do_cuda,
            args.batch, data_loaders, loss_fn )

        print( "Testing model on test data..." )
        test( model, args.new_weights_file, do_cuda, args.batch,
            data_loaders["test"], loss_fn )

    # Test model with given weights file
    elif args.pre_trained_weights_file:
        test( model, args.pre_trained_weights_file, do_cuda, args.batch,
            data_loaders["test"], loss_fn )

    else:
        print("No weights file specified. Ending...")


def train( model, epochs, weights_file, lr, do_cuda, batch_size, data_loaders,
        loss_fn ):
    """
    Train a model for a set number of epochs. Save model in weights_file
    """

    optimizer       = Adam( model.parameters(), lr=lr )
    full_loss_log   = []
    best_loss       = math.inf
    since           = time.time()

    # train for set number of epochs
    for epoch in range( epochs ):
        print( "Epoch: " + str( epoch ) )

        for phase in ["train", "val"]:
            
            epoch_loss, loss_log = do_epoch( model, data_loaders[phase], 
                                    loss_fn, optimizer, phase, do_cuda )
            full_loss_log += loss_log

            print('Epoch#{:d} {} Loss: {:.4f}'.format(epoch, phase, epoch_loss))

            # Save the best model checkpoint so far
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                save(best_model_wts, weights_file)


    # Print some stats
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    matplotlib.pyplot.plot(loss_log)
    matplotlib.pyplot.title('Loss')
    matplotlib.pyplot.savefig('Loss_'+str(epochs)+'.png')


def test( model, weights_file, do_cuda, batch_size, data_loader, loss_fn ):
    """
    Test a model loaded from weights file
    """

    model.load_state_dict( load( weights_file ) )
    model.eval()

    model_out_to_color_string_test( model, data_loader.dataset, do_cuda )

    epoch_loss, loss_log = do_epoch( model, data_loader, loss_fn, None, 
                                "test", do_cuda )


    print( 'Final Test Loss: {:.4f}'.format( epoch_loss ) )


def do_epoch( model, data_loader, loss_fn, optimizer, phase, do_cuda ):
    """
    Iterate over all samples in data_loader
    """

    running_loss    = 0.0
    total_samples   = 0
    loss_for_log    = 0.0
    loss_log        = []

    # Set model to train or eval mode
    if phase == 'train':
        model.train()  
    else:
        model.eval()   

    # Iterate over dataset
    for batch_idx, sample in enumerate( data_loader ):
        total_samples += 1
        if do_cuda:
            img         = Variable( sample[0].cuda() )
            target_hsv  = Variable( sample[1].cuda() )
        else:
            img         = Variable( sample[0] )
            target_hsv  = Variable( sample[1] )

        if phase == "train": optimizer.zero_grad()

        # Put image through model
        output_hsv = model(img)

        # Loss
        loss = loss_fn(output_hsv, target_hsv)

        # Sum up loss
        running_loss += loss.item()
        loss_for_log += loss.item()

        if phase == "train":
            # Backprop
            loss.backward()
            optimizer.step()

        # Print updates
        if batch_idx % 50 == 0:
            print( '(' + phase + ') Batch: ' + str(batch_idx) )
            print( '(' + phase + ') loss: ', loss.item() )
            loss_log.append( loss_for_log/50 )
            loss_for_log = 0

    #After epoch is done...
    epoch_loss = running_loss / total_samples

    return epoch_loss, loss_log


def model_out_to_color_string_test( model, dataset, do_cuda, num_samples=10 ):
    """
    Pick random color strings and compare to hsv of output
    """

    random.seed()
    accuracy = 0.0
    total_correct = 0
    hsv_scale = np.array([179.,255.,255.])

    for i in range( num_samples ):
        r = random.randint( 0, len(dataset) - 1 )
        img, target_hsv = dataset[r]

        # Convert to proper form
        if do_cuda:
            img = Variable(img.cuda())
        else:
            img = Variable(img)
        img = img.unsqueeze(0)

        output_hsv = model( img )
        output_hsv = output_hsv.detach().data.numpy()[0]
        color_string    = model_out_to_color_fn( output_hsv )
        scaled_out_hsv  = np.ceil( output_hsv * hsv_scale ) 
        scaled_tar_hsv  = np.ceil( target_hsv.detach().data.numpy() * hsv_scale )

        print( "Color String:     " + color_string )
        print( "HSV Value:        " + str( scaled_out_hsv ) )
        print( "Target HSV Value: " + str( scaled_tar_hsv ) )
        print()


def prep_data( batch_size, seed, data_dir ):
    """
    Prepare datasets and loaders using random seed
    """

    dataset = ProductDataset(
            data_dir=data_dir,
            download=False,
            image_transform=transforms.Compose([
                            transforms.Resize( INPUT_SIZE ),
                            transforms.CenterCrop( INPUT_SIZE ),
                            transforms.ToTensor(),
                            transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )]),
            color_string_to_hsv_fn=color_to_hsv_fn
            )
    print()

    # Split into train and validation 
    data_size   = len( dataset )
    val_split   = 0.15
    test_split  = 0.05
    shuffle     = True
    indices     = list( range( data_size ) )
    split_val   = int( np.floor( val_split * data_size ) )
    split_test  = int( np.floor( test_split * data_size) )
    if shuffle :
        np.random.seed( seed )
        np.random.shuffle( indices )
    train_indices   = indices[split_test + split_val:]
    val_indices     = indices[split_test: split_val]
    test_indices    = indices[:split_test]

    print("Train samples: " + str( len( train_indices ) ) )
    print("Val samples:   " + str( len( val_indices ) ) )
    print("Test samples:  " + str( len( test_indices ) ) )

    # Creating PT data samplers and loaders:
    train_sampler   = SubsetRandomSampler( train_indices )
    valid_sampler   = SubsetRandomSampler( val_indices )
    test_sampler    = SubsetRandomSampler( test_indices )

    data_loaders = {}
    data_loaders["train"]   = DataLoader( dataset, batch_size=batch_size, 
                                    sampler=train_sampler )
    data_loaders["val"]     = DataLoader( dataset, batch_size=batch_size,
                                    sampler=valid_sampler )
    data_loaders["test"]    = DataLoader( dataset, batch_size=batch_size,
                                    sampler=test_sampler )

    return data_loaders
    


if __name__ == '__main__':
    main()
