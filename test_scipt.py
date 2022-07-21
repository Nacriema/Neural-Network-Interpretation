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
from torchvis import VanillaBackprop, DeconvNet, GradCAM, GuidedBackprop


if __name__ == '__main__':
    # Specify the path to the saved model, if needed
    os.environ['TORCH_HOME'] = '/media/hp/01D576CEBCC511F0/Pytorch/Neural-Network-Interpretation/models/vgg_16_pretrained'
    pretrained_model = models.vgg16(pretrained=True)

    # print("===== PRETRAINED MODEL =====")
    # print(pretrained_model)

    image = Image.open('./input_images/cat_dog.png').convert('RGB')
    image_tensor = preprocess_image(image, resize_im=False)

    # # VANILLA BACKPROPAGATION
    # VBP = VanillaBackprop(pretrained_model, layer="features.0")
    #
    # # Generate gradients
    # vanilla_grads = VBP.generate_gradients(image_tensor, target_class=243)
    #
    # grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # save_class_activation_images(image, grayscale_vanilla_grads, file_name='result3')
    # print('Vanilla backprop completed !')

    # DECONV NET
    # DN = DeconvNet(pretrained_model, layer="features.0")
    #
    # # Generate gradients
    # grads = DN.generate_gradients(image_tensor, target_class=243)
    #
    # grayscale_grads = convert_to_grayscale(grads)
    # save_class_activation_images(image, grayscale_grads, file_name='result4')
    # print('DeconvNet backprop completed !')

    # GRAD CAM
    # GC = GradCAM(pretrained_model, layers=["features.28"])
    # cam = GC.generate_gradients(image_tensor, target_class=243)
    # save_class_activation_images(image, cam, file_name='grad_cam', colormap='jet')
    # print('GradCAM completed !!!')

    # GUIDED BACKPROP
    GB = GuidedBackprop(pretrained_model, layer="features.0")

    grads = GB.generate_gradients(image_tensor, target_class=243)
    grayscale_grads = convert_to_grayscale(grads)
    save_class_activation_images(image, grayscale_grads, file_name='guided_backprop', colormap='jet')
    print('Guided Backpropagation completed !!!')

