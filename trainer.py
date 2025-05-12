from PIL import Image
import numpy as np
import os
from model import myNN

imageFolder = "../dogDataset/Images"
breed_folders = sorted(os.listdir(imageFolder))
clean_breed_names = [folder_name.split('-')[1] for folder_name in breed_folders]
breed_to_label = {breed: idx for idx, breed in enumerate(clean_breed_names)}
label_to_breed = {v: k for k, v in breed_to_label.items()}

def load_images(folder , label , size = (128,128)):
    X = []
    Y = []

    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).resize(size).convert("RGB")
            img_array = np.array(img).reshape(-1) / 255.0
            
            X.append(img_array)
            Y.append(label)

    return X , Y

X_total = []
Y_total = []

for breed_folder in breed_folders :
    breed_name = '-'.join(breed_folder.split('-')[1:])
    label = breed_to_label[breed_name]
    folder_path = os.path.join(imageFolder , breed_folder)
    X , Y = load_images(folder_path , label)
    X_total.extend(X)
    Y_total.extend(Y)

X_total = np.array(X_total).T
Y_total = np.array(Y_total)

def one_hot(Y, num_classes):
    one_hot_Y = np.zeros((num_classes, len(Y)))
    one_hot_Y[Y, np.arange(len(Y))] = 1
    return one_hot_Y

Y_total_onehot = one_hot(Y_total, len(breed_folders))


