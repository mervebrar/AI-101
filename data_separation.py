import os
import shutil
import random

base_dir = "/Users/merveebrardemirel/Desktop/practice_datasets/catdogdata"

subdirs = ["train/cats", "train/dogs", "test/cats", "test/dogs", "val/cats", "val/dogs" ]

for subdir in subdirs:
    path = os.path.join(base_dir, subdir)
    os.makedirs(path, exist_ok=True)
print("New directories created!")

source_cats = "/Users/merveebrardemirel/Desktop/practice_datasets/rawdatacatdog/cats_set"
source_dogs = "/Users/merveebrardemirel/Desktop/practice_datasets/rawdatacatdog/dogs_set"

train_cats = "/Users/merveebrardemirel/Desktop/practice_datasets/catdogdata/train/cats"
train_dogs = "/Users/merveebrardemirel/Desktop/practice_datasets/catdogdata/train/dogs"
test_cats = "/Users/merveebrardemirel/Desktop/practice_datasets/catdogdata/test/cats"
test_dogs = "/Users/merveebrardemirel/Desktop/practice_datasets/catdogdata/test/dogs"
val_cats = "/Users/merveebrardemirel/Desktop/practice_datasets/catdogdata/val/cats"
val_dogs = "/Users/merveebrardemirel/Desktop/practice_datasets/catdogdata/val/dogs"

train_split_ratio = 0.7
test_split_ratio= 0.15
val_split_ratio=0.15


cat_images = os.listdir(source_cats)
random.shuffle(cat_images)

split_index_cats_train = int(len(cat_images) * train_split_ratio)
split_index_cats_test = int(len(cat_images) * test_split_ratio)
split_index_cats_val = int(len(cat_images) * val_split_ratio)
train_cat_images = cat_images[:split_index_cats_train]
test_cat_images = cat_images[:split_index_cats_test]
val_cat_images = cat_images[:split_index_cats_val]

for img in train_cat_images:
    shutil.copy(os.path.join(source_cats, img), train_cats)
for img in test_cat_images:
    shutil.copy(os.path.join(source_cats, img), test_cats)
for img in val_cat_images:
    shutil.copy(os.path.join(source_cats, img), val_cats)

dog_images = os.listdir(source_dogs)
random.shuffle(dog_images)

split_index_dogs_train = int(len(dog_images) * train_split_ratio)
split_index_dogs_test = int(len(dog_images) * test_split_ratio)
split_index_dogs_val = int(len(dog_images) * val_split_ratio)
train_dog_images = dog_images[:split_index_dogs_train]
test_dog_images = dog_images[:split_index_dogs_test]
val_dog_images = dog_images[:split_index_dogs_val]
for img in train_dog_images:
    shutil.copy(os.path.join(source_dogs, img), train_dogs)
for img in test_dog_images:
    shutil.copy(os.path.join(source_dogs, img), test_dogs)
for img in val_dog_images:
    shutil.copy(os.path.join(source_dogs, img), val_dogs)
print("Images are separated!")