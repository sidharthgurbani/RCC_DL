import pandas as pd
import numpy as np
from PIL import Image
import pickle


def load_image_path_from_file(file):
    df = pd.read_csv(file)
    data = df.to_numpy()
    X = data[:, 0]
    y = data[:,1]

    return X, y


def save_dataset_object(file):
    X, y = load_image_path_from_file(file=file)
    # save_data = np.zeros([X.shape[0], 2])
    save_data = []

    for i, x in enumerate(X):
        image = Image.open(x)
        center_w, center_h = [image.size[0] / 2, image.size[1] / 2]
        left = center_w - 256
        right = center_w + 256
        top = center_h - 256
        bottom = center_h + 256
        # print(left, top, right, bottom)
        im = image.crop((left, top, right, bottom))
        data = np.asarray(im).tolist()
        X[i] = data
        # print(type(data))
        save_data.append([data, y[i]])
        # save_data[i, 0] = data
        # save_data[i, 1] = y[i]

    print(type(save_data))
    with open("../data/dataset.obj", "wb") as output_file:
        pickle.dump(save_data, output_file)

    # filename = "data/dataset.npy"
    # np.save(filename, save_data)


def load_dataset():
    with open("../data/dataset.obj", "rb") as input_file:
        dataset = pickle.load(input_file)

    X = []
    y = []
    for data in dataset:
        X.append(data[0])
        y.append(data[1])

    return np.asarray(X), np.asarray(y)

    # return np.load(file=file, allow_pickle=True)