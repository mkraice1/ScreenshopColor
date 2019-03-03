import random
import os

test_dir = "test_data/"
train_dir = "train_data/"
val_dir = "val_data/"

random.seed()

for filename in os.listdir(test_dir):
	r = random.random()

	if r < .8:
		# Save to train
		os.rename(test_dir + filename, train_dir + filename)
	elif r < .95:
		# Save in val
		os.rename(test_dir + filename, val_dir + filename)
	else:
		# Save in test
		pass