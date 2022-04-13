#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Apr 09 21:01:54 2022

@author: Nacriema

Refs:

"""
import cv2
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from utils import random_transform, preprocess, denormalize, create_gauss_2d
from functools import reduce
from PIL import Image

os.environ['TORCH_HOME'] = "./models"


class SaveFeatures(object):
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


class FilterVisualizer(object):
    def __init__(self, size=56, upscaling_steps=2, upscaling_factor=1.2, model=None):
        self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        self.model = model.cuda().eval()
        self.output = None
        for p in self.model.parameters():
            p.requires_grad = False

    def _get_module_by_name(self, access_string):
        names = access_string.split(sep='.')
        return reduce(getattr, names, self.model)

    def visualize(self, layer, filter, use_one_unit=False, lr=0.1, opt_steps=10, angle=5, blur=None, use_rotate=None):
        sz = self.size
        img = np.uint8(np.random.uniform(0, 255, (sz, sz, 3)))  # np array (H, W, C) with ints values for each pixel
        # activations = SaveFeatures(list(self.model.children())[0][layer])
        activations = SaveFeatures(self._get_module_by_name(layer))

        for _ in range(self.upscaling_steps):
            print("New upscaled step !!!")
            img_var = preprocess(img)  # img_var is Tensor (C, H, W)
            img_var = torch.unsqueeze(img_var, 0)
            img_var = img_var.cuda()
            img_var.requires_grad_()

            optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)
            for n in range(opt_steps):
                optimizer.zero_grad()
                self.model(img_var)
                if use_one_unit:
                    activations_size_x, activations_size_y = activations.features[0, filter].size()
                    # weights = create_gauss_2d(activations_size_x, activations_size_y)
                    # loss = - torch.mul(activations.features[0, filter], weights)
                    loss = - activations.features[0, filter][activations_size_x // 2, activations_size_y // 2]
                else:
                    loss = - activations.features[0, filter].mean()
                print(f"Loss = {loss}")
                loss.backward()
                optimizer.step()
            # Convert image back to the form of numpy and (H, W, C)
            img_var = denormalize(img_var[0])
            img = np.uint8(img_var.data.cpu().numpy().transpose(1, 2, 0) * 255)

            self.output = img

            sz = int(self.upscaling_factor * sz)
            img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)

            # Manipulate the rotation here !!!
            if use_rotate:
                img = random_transform(img, angle)

            # Use bilateral filter instead to preserve edge sharp
            if blur is not None:
                img = cv2.bilateralFilter(img, blur, 75, 75)

        self.save(layer, filter)
        activations.close()

    def save(self, layer, filter):
        plt.imsave("./result_images/" + str(self.model.__class__.__name__) + '_layer_' + layer.replace('.', '_') + '_' + str(filter) + ".jpg", np.clip(self.output, 0, 255))

    def plot_image(self, layer, filter):
        img = Image.open("./result_images/" + str(self.model.__class__.__name__) + '_layer_' + layer.replace('.', '_') + '_' + str(filter) + ".jpg")
        plt.figure(figsize=(7, 7))
        plt.imshow(img)
        plt.show()
