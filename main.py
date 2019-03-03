from product_dataset import ProductDataset, color_to_hsv_fn, ImageToHsvNet
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
import numpy as np
import argparse


def main():

	# Comman line interface
    parser = argparse.ArgumentParser(description='Regression color img to hsv')
    parser.add_argument('--load', dest='pre_trained_weights_file',
                        help='path to weights file to be loaded')
    parser.add_argument('--save', dest='new_weights_file',
                        help='path to save the weights file')
    parser.add_argument('--cuda', dest='cuda', default="False",
                        help='If set to false, will not use GPU. defaults to False')
    parser.add_argument('--epochs', dest='epochs', default=5, type=int,
                        help='Specify the number of epochs for training')
    parser.add_argument('--batch', dest='batch', default=8, type=int,
                        help='Batch size when training')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float,
                        help='Learning rate')
    args = parser.parse_args()

    # Using cuda?
    if args.cuda == "True":
        do_cuda = True
        print("Using GPU")
    else:
        do_cuda = False

    print (args)


    #Set model
    model = ImageToHsvNet()


    if args.new_weights_file:
    	train(model, args.epochs, args.new_weights_file, args.lr, do_cuda, args.batch)



def train(model, epochs, weights_file, lr, do_cuda, batch_size):

	# Load dataset
	dataset = ProductDataset(
            data_dir='./train_data',
            download=False,
            image_transform=lambda pil_im: transforms.ToTensor()(pil_im),
            color_string_to_hsv_fn=color_to_hsv_fn
            )

	# Split into train and validation 
	seed 		= 42
	val_split 	= 0.15
	test_split 	= 0.05
	shuffle 	= True
	data_size 	= 10000
	indices 	= list(range(data_size))
	split_val 	= int(np.floor(val_split * data_size))
	split_test 	= int(np.floor(test_split * data_size))
	if shuffle :
	    np.random.seed(seed)
	    np.random.shuffle(indices)
	train_indices 	= indices[split_test + split_val:]
	val_indices		= indices[split_test: split_val]
	test_indices 	= indices[:split_test]

	# Creating PT data samplers and loaders:
	train_sampler 	= SubsetRandomSampler(train_indices)
	valid_sampler 	= SubsetRandomSampler(val_indices)
	test_sampler 	= SubsetRandomSampler(test_indices)


	train_loader 		= DataLoader(dataset, batch_size=batch_size, 
                                    sampler=train_sampler)
	validation_loader 	= DataLoader(dataset, batch_size=batch_size,
                                    sampler=valid_sampler)
	test_loader 		= DataLoader(dataset, batch_size=batch_size,
                                    sampler=test_sampler)


	loss_fn 	= BCEWithLogitsLoss()
	optimizer 	= Adam( model.parameters(), lr=lr )


	# train for set number of epochs
	for epoch in range( epochs ):
        print( "epoch: " + str(epoch) )
        for batch_idx, sample in enumerate( train_loader ):
        	if do_cuda:
                img 		= Variable( sample[0].cuda() )
                target_hsv 	= Variable( sample[1].cuda() )
            else:
                img 		= Variable( sample[0] )
                target_hsv 	= Variable( sample[1] )

            optimizer.zero_grad()

            # Put image through model
            output_hsv = model(img)

            # Loss
            loss = loss_fn(output_hsv, target_hsv)

            # Backprop
            loss.backward()
        	optimizer.step()

        	if batch_idx % 100 == 0:
        		print( loss.data[0] )

	



if __name__ == '__main__':
	main()
