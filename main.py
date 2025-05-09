import os
import kagglehub
import tensorflow as tf
import torch
import torch.nn as nn  
import torch.nn.functional as F
import shutil
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tf.keras import layers,models

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resizing to a standard size
    transforms.ToTensor(),          # convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize
])

path = "/media/biprarshi/common/files/AI/kagglehub_cache/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/versions/5"
animal_path = path + "/animals/animals"

with open(path + "/name of the animals.txt") as f:
    classes = f.read().split('\n')

num_classes = len(classes)

def vectorize_objects(path, cache_file="vectorized_animals.pt"):
    if os.path.exists(cache_file):
        print(f"Loading cached vectorized data from {cache_file}")
        return torch.load(cache_file)
    
    vectorized_animals = {}
    for animal in classes :
        animal_folder = os.path.join(path, animal)
        if not os.path.isdir(animal_folder):
            print(f"Skipping missing folder for {animal}")
            continue
        
        images = []
        for img_file in os.listdir(animal_folder) :
            img_path = os.path.join(animal_folder, img_file)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')  # ensure 3 channels
                    tensor_img = transform(img)
                    images.append(tensor_img)
            except Exception as e:
                print(f"Could not process {img_path}: {e}")
        
        vectorized_animals[animal] = images
    torch.save(vectorized_animals, cache_file)
    print(f"Saved vectorized data to {cache_file}")
    return vectorized_animals

def split_animals(vectorized_animals , train_ratio = 0.8) :
    train_set = {}
    test_set = {}
    for animal,images in vectorized_animals.items() :
        num_train = int(len(images) * train_ratio)
        train_set[animal] = images[:num_train]
        test_set[animal] = images[num_train:]
    return train_set, test_set

train_set, test_set = split_animals(vectorize_objects(animal_path))

train_loader = DataLoader(
    dataset = train_set,
    batch_size = 32,
    shuffle = True
)

test_loader = DataLoader(
    dataset = test_set,
    batch_size = 32,
    shuffle = False
)

class AnimalCNN(nn.Module) :
    def __init__(self, num_classes) :
        super(AnimalCNN , self).__init__()
        