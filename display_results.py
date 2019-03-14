"""
Script to load a model and see some results.
Will display images and the final closest matching color string

"""
from product_dataset import ProductDataset, color_to_hsv_fn, model_out_to_color_fn
from torchvision.models import alexnet, resnet18, resnet34
from torch.nn import Linear
from torch.autograd import Variable



from torchvision import transforms
from torch import load

from PIL import Image, ImageDraw


import argparse
import random
import math

INPUT_SIZE = 224


def main():

    # Comman line interface
    parser = argparse.ArgumentParser(description='Regression color img to hsv')
    parser.add_argument('--load', dest='weights_file',
                        help='Path to weights file to be loaded. If specified, will train model.')
    parser.add_argument('--data-dir', dest='data_dir', default="./good_data",
                    help='Path to data (default: ./good_data).')
    parser.add_argument('--model', dest='model_type', default="alexnet",
                    help='Type of model (default: alexnet): alexnet, resnet18, resnet34')
    parser.add_argument('--num-samples', dest='num_samples', default=8, type=int,
                    help='Number of samples to test against')
    args = parser.parse_args()

    # Load dataset
    dataset = ProductDataset(
        data_dir=args.data_dir,
        download=False,
        image_transform=transforms.Compose([
                        transforms.Resize( INPUT_SIZE ),
                        transforms.CenterCrop( INPUT_SIZE ),
                        transforms.ToTensor(),
                        transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )]),
        color_string_to_hsv_fn=color_to_hsv_fn
        )

    # Load data and models
    num_outputs = 3
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

    model.load_state_dict( load( args.weights_file ) )
    model.eval()


    # Go through samples and display results
    random.seed()
    rand_i = random.sample(range(len(dataset)), args.num_samples)
    thumb_size = 200
    x_pos = 0
    y_pos = 0
    imgs_in_row = 6

    w = imgs_in_row * thumb_size
    h = math.ceil(thumb_size * args.num_samples / imgs_in_row)
    results_im = Image.new('RGB', (w, h))

    for i in rand_i:
        img, target_hsv = dataset[i]
        disp_fp = args.data_dir + "/" + str(i) + "_img.jpg"

        # Convert to proper form
        img = Variable(img).unsqueeze(0)

        output_hsv = model( img )
        output_hsv = output_hsv.detach().data.numpy()[0]
        color_string    = model_out_to_color_fn( output_hsv )

        # Go to next row in image
        if x_pos / thumb_size == imgs_in_row:
            x_pos = 0
            y_pos += thumb_size

        disp_img = Image.open(disp_fp)
        disp_img.thumbnail((thumb_size, thumb_size))

        d = ImageDraw.Draw(disp_img)
        d.text((10,20), color_string, fill=(255,255,0))

        results_im.paste( disp_img, (x_pos, y_pos) )
        x_pos += thumb_size

    results_im.save("some_results.jpg")
    results_im.show()



if __name__ == "__main__":
    main()