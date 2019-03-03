import os
import json
import csv
import re


re_only_chars = re.compile('[^a-z]')
directory = "test_data/"
color_dict = dict()

pink_set = ("pink", "rose")
red_set = ("red", "blush", "magenta", "berry")
maroon_set = ("maroon", "burgundy")
orange_set = ("orange", "rust", "apricot", "peach", "salmon", "copper",
			"bronze", "coral", "jacinth")
yellow_set = ("yellow", "gold", "champagne", "lemon", "mustard")
green_set = ("green", "lime", "olive", "hunter", "army", "camo", "forest")
teal_set = ("teal", "aqua", "cyan", "mint", "turquoise")
blue_set = ("blue", "indigo", "navy", "royal", "sapphire", "denim", "sky",
			"sea")
purple_set = ("purple", "violet", "wine", "plum", "mauve", "lilac", "lavender",
			"fuchsia")
white_set = ("white", "ivory", "cream", "bone")
black_set = ("black",)
gray_set = ("gray", "grey", "silver", "stone", "heather", "charcoal", "dark",
			"graphite", "cobalt")
tan_set = ("tan", "beige", "khaki", "camel", "sand", "taupe", "nude",)
brown_set = ("brown", "coffee", "espresso", "leather", "toffee", "mocha", "choco")
multi_set = ("multi", "mix", "colorful")
none_set = ("none", "nan")

normal_colors = {"pink": pink_set, "red": red_set, "maroon": maroon_set, "orange": orange_set, 
				"yellow": yellow_set, "green": green_set, "teal": teal_set, "blue": blue_set,  
				"purple": purple_set, "white": white_set, "black": black_set, 
				"gray": gray_set, "tan": tan_set, "brown": brown_set, "multi": multi_set, 
				"none": none_set}


def main():
	others = 0
	# Go through downloaded data and organize colors
	for filename in os.listdir(directory):
		if filename.endswith(".json"):
			json_data=open(directory + filename).read()
			data = json.loads(json_data)

			# Lower and remove non-chars
			refined_color = clean_color_string(data["raw_color"])

			# Cluster color strings
			color_found = False
			for normal_color, color_set in normal_colors.items():
				for color in color_set:
					if color in refined_color:
						color_found = True
						safe_insert(color_dict, normal_color)
						break
				if color_found:
					break
			
			# No color found
			if not color_found:
				others += 1
				safe_insert(color_dict, refined_color)
			
	print(others)

	# Dump to csv to take a look
	with open("color_strings.csv", "w", newline="") as csv_file:
		writer = csv.writer(csv_file, delimiter=",")
		for key, val in color_dict.items():
			writer.writerow([key, val])


def safe_insert(d, val):
	if val in d:
		d[val] += 1
	else:
		d[val] = 1

# Function to get rid of non lower case chars
def clean_color_string(s):
	return re_only_chars.sub(' ', s.lower()).strip()


if __name__ == "__main__":
	main()
