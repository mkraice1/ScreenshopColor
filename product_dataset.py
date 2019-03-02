import torch
from torch import nn
from torch import optim

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from google.cloud import bigquery, client

import requests
from io import BytesIO
from PIL import Image

import numpy as np
import os
import json

# gpu available?
device = "cuda" if torch.cuda.is_available() else "cpu"

def color_to_hsv_fn(color_string: str) -> np.array:
    print(color_string)


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
        """
        return torch.tensor(hsv).div(255.)

    def get_image(self, index):
        """"
        load an image if downloaded, else get from its url
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
        product_info = self.get_product_info(index)
        color_string = product_info['raw_color']
        image = self.get_image(index)
        print(color_string)
        color_hsv = self.color_string_to_hsv_fn(color_string)

        inputs = self.image_transform(image)
        targets = self.hsv_transform(color_hsv)

        return inputs, targets

    def __len__(self):
        return 10000


if __name__ == '__main__':
    dataset = ProductDataset(
            data_dir='./test_data',
            download=False,
            image_transform=lambda pil_im: transforms.ToTensor()(pil_im),
            color_string_to_hsv_fn=color_to_hsv_fn
            )

    img, color_hsv = dataset[1]

