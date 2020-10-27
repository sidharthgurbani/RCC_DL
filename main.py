from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from load_dataset import load_dataset
from extract_images import generate_images_from_dcm
from train_model import train_model


def run_model():
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    dataset = load_dataset("data/dataset.npy")
    X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, random_state=42)
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    input_data = {
        'train': [X_train, y_train],
        'test': [X_test, y_test]
    }

    model_conv = torchvision.models.resnet18(pretrained=True)

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    print("features {}".format(num_ftrs))
    print(model_conv)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.00005, weight_decay=0.05)

    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_conv, 'min', patience=2,
                                                            verbose=True, factor=0.2)

    model_ft = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, device,
                           input_data=input_data, num_epochs=10)


def main():
    # dataset_dir = "../RCC Portal Venous DICOMs/Corrected RCCPV Datasets/"
    # generate_images_from_dcm(dataset_dir=dataset_dir)
    # dataset = load_dataset("data/dataset.npy")
    # print(type(dataset))
    run_model()

    return


# import sys
# print(sys.version)
#
# import numpy
# print(numpy.__version__)
main()
# x = sorted(having_rt_struct_folder("../Dataset/RCC Portal Venous DICOMs/Corrected RCCPV Datasets/"))
# df1 = pd.DataFrame(x)
# df1.to_excel("Folders_Having_RTSTRUCT.xlsx")
# print(x)
# readImage()