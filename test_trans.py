import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms

#image_url = "https://gloimg.drlcdn.com/L/pdm-product-pic/Clothing/2016/12/27/goods-img/1515700377016972712.png"
#response = requests.get(image_url)
#img = Image.open(BytesIO(response.content))

trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

fp = "good_data/3538_img.jpg"
img = Image.open(fp)

new_img = trans(img)

print(new_img.shape)