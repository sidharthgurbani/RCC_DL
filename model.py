import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy


def train_model(model, criterion, optimizer, scheduler, device, input_data, num_epochs):
    phase = 'train'

    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_error = 2.0
    best_model_wts1 = model.state_dict()

    dataset_sizes = {p: len(input_data[phase]) for p in ['train', 'test']}


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        k = 0
        running_loss = 0.0
        running_corrects = 0

        if phase == 'train':
            print(phase)
            model.train()
            for inputs, labels in input_data[phase]:
                k += 1
                if (k % 100 == 0):
                    print(k / 100)

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = (running_loss * 1.0) / dataset_sizes[phase]
            epoch_acc = (running_corrects.double()) / dataset_sizes[phase]


        if phase == 'test':
            model.eval()
            cn = 0
            run_cor = 0

            for inputs, labels in input_data[phase]:
                k += 1
                if (k % 100 == 0):
                    print(k / 100)

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = (running_loss * 1.0) / dataset_sizes[phase]
            epoch_acc = (running_corrects.double()) / dataset_sizes[phase]

            scheduler.step(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'test' and epoch_loss < best_error:
                best_error = epoch_loss
                best_model_wts1 = copy.deepcopy(model.state_dict())

            print('Best val Acc: {:4f}'.format(best_acc))
            print('Least val Err: {:4f}'.format(best_error))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model