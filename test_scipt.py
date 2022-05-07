#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 06 17:36:35 2022

@author: Nacriema

Refs:

"""
import os
from torchvision import models
from PIL import Image
from torchvis.utils.image import preprocess_image, convert_to_grayscale, save_class_activation_images
from torchvis import VanillaBackprop

if __name__ == '__main__':
    # Specify the path to the saved model, if needed
    os.environ['TORCH_HOME'] = '/media/hp/01D576CEBCC511F0/Pytorch/Neural-Network-Interpretation/models/vgg_16_pretrained'
    pretrained_model = models.vgg16(pretrained=True)

    image = Image.open('./input_images/cat_dog.png').convert('RGB')
    image_tensor = preprocess_image(image, resize_im=False)
    VBP = VanillaBackprop(pretrained_model, layer="features.0")

    # Generate gradients
    vanilla_grads = VBP.generate_gradients(image_tensor, target_class=243)

    print('xxx')
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    save_class_activation_images(image, grayscale_vanilla_grads, file_name='result3')
    print('Vanilla backprop completed !')


