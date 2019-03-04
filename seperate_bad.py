import random
from shutil import copyfile
import os
import numpy as np
import json
import re


# Nones: 		2143
# Not founds: 	831

re_only_chars = re.compile('[^a-z]')
test_dir = "test_data/"
good_dir = "good_data/"
bad_dir = "bad_data/"
move_files = True

def main():
	nones = 0
	not_found = 0
	num_good = 0

	for filename in os.listdir("test_data/"):
		if filename.endswith(".json"):
			json_data=open(test_dir + filename).read()
			data = json.loads(json_data)
			nice_string = re_only_chars.sub(" ", data["raw_color"].lower()).strip()
			found = False
			is_none = False

			for word in nice_string.split():
				if word in ["none", "nan"]:
					is_none = True
					nones += 1
					break
				elif word in color_table:
					found = True
					break


			if not found and not is_none:
				not_found += 1

			if move_files:
				if found:
					new_file = str(num_good) + "_info.json"
					copyfile(test_dir + filename, good_dir + new_file)
					num_good += 1
				else:
					copyfile(test_dir + filename, bad_dir + filename)


	print("Nones: " + str(nones))
	print("Not founds: " + str(not_found))




color_table = {
                "pink": np.array([172, 242, 196]),
                "rose": np.array([172, 242, 196]),
                "blush": np.array([172, 242, 196]),
                "red": np.array([0,255,255]),
                "magenta": np.array([0,255,255]),
                "berry": np.array([0,255,255]),
                "maroon": np.array([0,0,122]),
                "burgundy": np.array([0,0,122]),
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

if __name__ == "__main__":
	main()
