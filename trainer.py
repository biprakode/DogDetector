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

image_paths = []
labels = []

for breed_folder in breed_folders:
    breed_name = '-'.join(breed_folder.split('-')[1:])
    label = breed_to_label[breed_name]
    folder_path = os.path.join(imageFolder, breed_folder)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_paths.append(os.path.join(folder_path, filename))
            labels.append(label)


def one_hot(Y, num_classes):
    one_hot_Y = np.zeros((num_classes, len(Y)))
    one_hot_Y[Y, np.arange(len(Y))] = 1
    return one_hot_Y


def load_batch(image_paths, labels, batch_size, start_idx, size=(128, 128)):
    X_batch = []
    Y_batch = []
    end_idx = min(start_idx + batch_size, len(image_paths))

    for i in range(start_idx, end_idx):
        img = Image.open(image_paths[i]).resize(size).convert("RGB")
        img_array = np.array(img).reshape(-1) / 255.0
        X_batch.append(img_array)
        Y_batch.append(labels[i])

    X_batch = np.array(X_batch).T
    Y_batch = np.array(Y_batch)

    return X_batch, Y_batch

def save_parameters(parameters, filename="model_parameters.npz"):
    np.savez(filename, **parameters)

def load_parameters(filename="model_parameters.npz"):
    data = np.load(filename)
    return {key: data[key] for key in data}

model = myNN(layer_dims=[(128 * 128 * 3), 128 , len(breed_folders)] , learning_rate=0.00001 , seed = 42 )

batch_size = 64
epochs = 10
num_batches = len(image_paths)//batch_size

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    epoch_cost = 0

    for batch in range(num_batches):
        # Shuffle data at the start of each epoch
        perm = np.random.permutation(len(image_paths))
        image_paths = [image_paths[i] for i in perm]
        labels = [labels[i] for i in perm]

        X_batch, Y_batch = load_batch(image_paths, labels, batch_size, batch * batch_size)
        Y_batch_onehot = one_hot(Y_batch, len(breed_folders),)
        cost = model.train_batch(X_batch, Y_batch_onehot)
        epoch_cost += cost

        if batch % 10 == 0:
            print(f"Batch {batch}/{num_batches} — Cost: {cost}")

    print(f"Epoch {epoch+1} completed — Average Cost: {epoch_cost/num_batches}")
    save_parameters(model.parameters)
    model.parameters = load_parameters()