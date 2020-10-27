import pandas as pd
import numpy as np
from PIL import Image


def load_image_path_from_file(file):
    df = pd.read_csv(file)
    data = df.to_numpy()
    X = data[:, 0]
    y = data[:,1]

    return X, y


def save_dataset_object(file):
    X, y = load_image_path_from_file(file=file)
    save_data = np.zeros([X.shape[0], 2], dtype=object)

    for i, x in enumerate(X):
        image = Image.open(x)
        center_w, center_h = [image.size[0] / 2, image.size[1] / 2]
        left = center_w - 256
        right = center_w + 256
        top = center_h - 256
        bottom = center_h + 256
        # print(left, top, right, bottom)
        im = image.crop((left, top, right, bottom))
        data = np.asarray(im)
        X[i] = data
        print(type(data))
        save_data[i, 0] = data
        save_data[i, 1] = y[i]

    print(type(save_data))
    # filename = "data/dataset.npy"
    # np.save(filename, save_data)


def load_dataset(file):
    return np.load(file=file, allow_pickle=True)