import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from google.cloud import bigquery, client

import requests
from io import BytesIO
from PIL import Image

import numpy as np
import os
import json
import re
import math

# gpu available?
device = "cuda" if torch.cuda.is_available() else "cpu"

# Variables for string to hsv
re_only_chars = re.compile('[^a-z]')
directory = "good_data/"

# Possiibly need if this is a multi-class problem?
# hsv_array = np.array([
#                     np.array([172, 242, 196]),
#                     np.array([0,255,255]),
#                     np.array([15,255,255]),
#                     np.array([27,255,255]),
#                     np.array([50,255,255]),
#                     np.array([85,255,255]),
#                     np.array([120,255,255]),
#                     np.array([120,255,122]),
#                     np.array([150,255,255]),
#                     np.array([0,0,255]),
#                     np.array([0,0,0]),
#                     np.array([0,0,122]),
#                     np.array([15,153,204]),
#                     np.array([15,229,122])
#                     ])
color_table = {
                "pink": np.array([172, 242, 196]),
                "rose": np.array([172, 242, 196]),
                "blush": np.array([172, 242, 196]),
                "red": np.array([179,255,255]),
                "magenta": np.array([179,255,255]),
                "berry": np.array([179,255,255]),
                "maroon": np.array([179,0,122]),
                "burgundy": np.array([179,0,122]),
                "orange": np.array([15,255,255]),
                "rust": np.array([15,255,255]),
                "apricot": np.array([15,255,255]),
                "peach": np.array([15,255,255]),
                "salmon": np.array([15,255,255]),
                "copper": np.array([15,255,255]),
                "bronze": np.array([15,255,255]),
                "coral": np.array([15,255,255]),
                "jacinth": np.array([15,255,255]),
                "yellow": np.array([27,255,255]),
                "gold": np.array([27,255,255]),
                "champagne": np.array([27,255,255]),
                "lemon": np.array([27,255,255]),
                "mustard": np.array([27,255,255]),
                "green": np.array([50,255,255]), 
                "lime": np.array([50,255,255]),
                "olive": np.array([50,255,255]),
                "hunter": np.array([50,255,255]),
                "army": np.array([50,255,255]),
                "camo": np.array([50,255,255]),
                "forest": np.array([50,255,255]),
                "teal": np.array([85,255,255]), 
                "aqua": np.array([85,255,255]),
                "aquamarine": np.array([85,255,255]),
                "cyan": np.array([85,255,255]),
                "mint": np.array([85,255,255]),
                "turquoise": np.array([85,255,255]),
                "blue": np.array([120,255,255]), 
                "indigo": np.array([120,255,255]),
                "royal": np.array([120,255,255]),
                "sapphire": np.array([120,255,255]),
                "denim": np.array([120,255,255]),
                "sky": np.array([120,255,255]),
                "sea": np.array([120,255,255]),
                "navy": np.array([120,255,122]),
                "purple": np.array([150,255,255]), 
                "violet": np.array([150,255,255]),
                "wine": np.array([150,255,255]),
                "plum": np.array([150,255,255]),
                "mauve": np.array([150,255,255]),
                "lilac": np.array([150,255,255]),
                "lavender": np.array([150,255,255]),
                "fuchsia": np.array([150,255,255]),
                "white": np.array([0,0,255]), 
                "ivory": np.array([0,0,255]),
                "cream": np.array([0,0,255]),
                "bone": np.array([0,0,255]),
                "black": np.array([0,0,0]), 
                "gray": np.array([0,0,122]), 
                "grey": np.array([0,0,122]),
                "charcoal": np.array([0,0,122]),
                "silver": np.array([0,0,122]),
                "stone": np.array([0,0,122]),
                "heather": np.array([0,0,122]),
                "graphite": np.array([0,0,122]),
                "cobalt": np.array([0,0,122]),
                "tan": np.array([15,153,204]), 
                "beige": np.array([15,153,204]),
                "khaki": np.array([15,153,204]),
                "camel": np.array([15,153,204]),
                "sand": np.array([15,153,204]),
                "taupe": np.array([15,153,204]),
                "nude": np.array([15,153,204]),
                "brown": np.array([15,229,122]),
                "coffee": np.array([15,229,122]),
                "espresso": np.array([15,229,122]),
                "leather": np.array([15,229,122]),
                "toffee": np.array([15,229,122]),
                "mocha": np.array([15,229,122]),
                "chocolate": np.array([15,229,122]),
                }


# def hsv_to_model_out( hsv ):
#     pass

# Convert color string to hsv value
def color_to_hsv_fn(color_string: str) -> np.array:
    nice_string = re_only_chars.sub(" ", color_string.lower()).strip()

    for word in nice_string.split():
        if word in color_table:
            return color_table[word]

    # Don't want None
    return np.array([0,0,0])


# Convert model output to closest color string
def model_out_to_color_fn( model_hsv ):
    hsv_scale   = np.array([179.,255.,255.])
    best_diff   = math.inf
    best_color  = ""


    for color, hsv in color_table.items():
        hsv_norm = hsv / hsv_scale
        diff = np.linalg.norm( model_hsv - hsv_norm )

        if diff < best_diff:
            best_diff   = diff
            best_color  = color

    return best_color

# Convert model output to closest color string
# def model_class_vector_to_color_fn( model_hsv ):
#     color_i = np.argmax( model_hsv )
#     return hsv_array[color_i]


class ProductDataset(Dataset):
    """
    dataset of 10k products aggregated from multiple vendors,
    with the following metadata:
        + image_url (str): url of a picture of the product
        + raw_color (str): string of the color of the product as specified by the vendor.  this is not normalized or cleaned in any way
    """
    cli = bigquery.Client(project='manymoons-215635')

    QUERY =\
    """
    SELECT
        raw_color, image_url
    FROM
        porsche_data.variants
    LIMIT
        10000
    """

    to_tensor = transforms.ToTensor()

    def __init__(
            self,
            data_dir,
            download,
            image_transform,
            color_string_to_hsv_fn):
        """
        args:
            + data_dir (str): path to directory where dataset will download
            + download (bool): download dataset?  set True for first time, else
                               False if already downloaded
            + image_transform (function):
                function to transform PIL image into input to be passed to a
                pytorch model
            + color_string_to_hsv_fn (function):
                function to transform raw_color str into hsv format.
                will be fed into ProductDataset.hsv_transform to turn into
                target for the output of a pytorch model
        """
        self.data_dir = data_dir
        self.image_transform = image_transform
        self.color_string_to_hsv_fn = color_string_to_hsv_fn

        if download:
            results = self.cli.query(self.QUERY)
            for index, result in enumerate(results):
                print('downloading {} of {} products...'.format(index, 10000), end='\r')
                self.save_product_info(index, dict(result))

    def _info_path(self, index):
        return os.path.join(self.data_dir, str(index) + '_info.json')

    def _img_path(self, index):
        return os.path.join(self.data_dir, str(index) + '_img.jpg')

    def save_product_info(self, index, info):
        """
        get and save product info
        """
        fp = self._info_path(index)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(self._info_path(index), 'w') as fp:
            json.dump(info, fp)

    def get_product_info(self, index):
        """"
        load product info
        """
        with open(self._info_path(index), 'r') as fp:
            info = json.load(fp)
        return info

    @staticmethod
    def hsv_transform(hsv):
        """
        convert np int array [0, 255] output of color_string_to_hsv_fn to
        pytorch float tensor w vals in [0, 1]
        CORRECT FOR HUE VALUES 0-180
        """
        hsv_copy = np.copy(hsv)
        if hsv is not None:
            hsv_copy[0] = hsv_copy[0] * 255./179.
            return torch.tensor( hsv_copy, dtype=torch.float ).div( 255. )
        return None

    # @staticmethod
    # def hsv_to_class_vector(hsv):
    #     """
    #     Possibly use if multi-class problem
    #     """
    #     hsv_vector = np.zeros(len(hsv_array))
    #     myindex = -1
    #     for i in range( len(hsv_array) ):
    #         if np.array_equal( hsv_array[i], hsv ):
    #             myindex = i
    #             break
    #     hsv_vector[myindex] = 1.
    #     return torch.tensor( hsv_vector, dtype=torch.float )

    def get_image(self, index):
        """"
        load an image if downloaded, else get from its url
        Note: Weird error in __getitem__ when getting img from png url
            From url, will return non-3 channel img
            From file, will return 3 channel...
        """
        fp = self._img_path(index)
        if os.path.exists(fp):
            img = Image.open(fp)
        else:
            url = self.get_product_info(index)['image_url']
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            img.convert('RGB').save(fp)
        return img

    def __getitem__(self, index):
        product_info    = self.get_product_info(index)
        color_string    = product_info['raw_color']
        image           = self.get_image(index)
        width, height   = image.size
        color_hsv       = self.color_string_to_hsv_fn(color_string)
        inputs          = self.image_transform(image)
        targets         = self.hsv_transform(color_hsv)
        #targets         = self.hsv_to_class_vector(color_hsv)
        # if inputs.shape[0] != 3:
        #     print(index)
        #     print(product_info["image_url"])
        return inputs, targets

    # NOT GREAT
    def __len__(self):
        return 6987


if __name__ == '__main__':
    input_size = 224
    dataset = ProductDataset(
            data_dir='./good_data',
            download=False,
            image_transform=transforms.Compose([
                            transforms.Resize(input_size),
                            transforms.CenterCrop(input_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            color_string_to_hsv_fn=color_to_hsv_fn
            )

    img, color_hsv = dataset[2]
    # Run first to dl all imgs
    # for i in range(len(dataset)):
    #     if i % 100 == 0:
    #         print( i )
    #     try:
    #         img, color_hsv = dataset[i]
    #     except Exception as e:
    #         print(e)
    #         print( "Bad i: " + str(i) )