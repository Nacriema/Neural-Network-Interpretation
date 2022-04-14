#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 14 17:22:50 2022

@author: Nacriema

Refs:

"""
import os
from torchvision import transforms, models
from PIL import Image
import torch


os.environ["TORCH_HOME"] = '../models/vgg_19_pretrained'
vgg19 = models.vgg19(pretrained=True)

# Transformation block
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open("./images/goldFish/layer_classifier_61.jpg")
# img = Image.open("./images/umbrela/layer_classifier_6879.jpg")
img.show()

img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)
vgg19.eval()
out = vgg19(batch_t)

with open('./imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

print(f"Prediction of {vgg19.__class__.__name__} model \nClass: {labels[index[0]]} \nPercentage: {percentage[index[0]].item()}")
