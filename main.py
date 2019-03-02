from product_dataset import ProductDataset, color_to_hsv_fn

def main():
	dataset = ProductDataset(
            data_dir='./test_data',
            download=False,
            image_transform=lambda pil_im: transforms.ToTensor()(pil_im),
            color_string_to_hsv_fn=color_to_hsv_fn
            )

	img, color_hsv = dataset[0]



if __name__ == '__main__':
	main()
