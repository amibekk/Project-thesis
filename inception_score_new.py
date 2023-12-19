'''
Code for the inception score from pytorch-ignite framework:
https://pytorch.org/ignite/generated/ignite.metrics.InceptionScore.html
'''
from collections import OrderedDict
import torch
from torch import nn
from ignite.engine import *
from ignite.utils import *

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

param_tensor = torch.zeros([1], requires_grad=True)
default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

def get_default_trainer():

    def train_step(engine, batch):
        return batch

    return Engine(train_step)

default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(4, 2)),
    ('fc', nn.Linear(2, 1))
]))

manual_seed(666)
''''
load and crop images in order to find inception score
'''
import os
import nibabel as nib
import numpy as np
images_path = "D:/Prosjektoppgave_Anna/Synthetic_images/bs16_epoch150/PyEnv3/Synthetic_images/bs16_150epochs_3nov"
#images_path = "D:/Prosjektoppgave_Anna/Val_bilder/PyEnv3/Real_images/Real_validation_data_bs16_for_epoch250"
#images_path = "D:/Prosjektoppgave_Anna/Training_datasett/PyEnv3/Real_images/Real_training_data_bs16_epoch"
#images_path = "D:/Prosjektoppgave_Anna/Synthetic_images/Larger_datasett_400/PyEnv3/Synthetic_images/bs16_150epochs_22_nov_larger_dataset"
images_list = os.listdir(images_path)
images = []
for file in images_list:
    image = nib.load(images_path + "/" + file).get_fdata()
    images.append(image)

np.random.shuffle(images)
images = images[0:100]
print(np.array(images).shape, ", # of images: ",len(images))

def preprocess_for_fid(images):
    images = np.array(images).astype('float32')
    images = torch.unsqueeze(torch.tensor(images), 1)
    if images.shape[1]:
        images = images.repeat(1, 3, 1, 1)
    #print(images.shape)
    return images

images = preprocess_for_fid(images)
print(np.array(images).shape, ", # of images: ",len(images))


from ignite.metrics import InceptionScore
import torch

metric = InceptionScore()
metric.attach(default_evaluator, "is")
state = default_evaluator.run([images])
print(state.metrics["is"])

