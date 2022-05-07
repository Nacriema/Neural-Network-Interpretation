#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 06 17:13:03 2022

@author: Nacriema

Refs:

"""
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import urllib.request
import io
import os
import copy
import torch
# This is deprecated

# Something need to be removed
import matplotlib.pyplot as plt
from .logger import print_info


def apply_heatmap(R: object, sx: float, sy: float) -> object:
    """
    This is a sensitive heatmap, used for the LRP.
    Consider to be removed or changed later !!!
    :param R:
    :param sx:
    :param sy:
    :return:
    """
    b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))
    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    heat_map = plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
    return heat_map


def convert_to_grayscale(im_as_arr) -> object:
    """
    Converts 3d image to grayscale
    :param im_as_arr:  array shape (D, H, W)
    :return:  grayscale_im (np array): Grayscale image with shape (1, H, W)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def read_image_url(image_url):
    """
    Read image as PIL from image url
    :param image_url: url for the image
    :return: PIL image instance
    """
    with urllib.request.urlopen(image_url) as url:
        image_file = io.BytesIO(url.read())
        im = Image.open(image_file)
        return im


def format_np_output(np_arr):
    """
    Convert all ouput to the same format with is 3 * H * W
    :param np_arr:
    :return:
    """
    # Np arr only has 2 dimensions
    # Result: add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)

    # Np arr has only one channel (assuming first dim is channel)
    # Result: Repeat first channel anh convert from 1 * H * W to 3 * H * W

    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)

    # Np arr is not of shape 3 * W * H
    # Result: Convert it to W * H * 3 in order to make it savable in PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose((1, 2, 0))

    # Np arr is normalized between 0 - 1
    # Result: Multiply with 255 and change type to make it savable vy PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
    Save numpy matrix of PIL
    :param im:
    :param path:
    :return:
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    print_info(f'Saving image to path: {path}....')
    im.save(path)


def save_gradient_images(gradient, file_name):
    """
    Exports the original gradient images
    :param gradient:
    :param file_name:
    :return:
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')
        # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join('./results', file_name + '.jpg')
    save_image(gradient, path_to_file)


def preprocess_image(pil_im, resize_im=True):
    """
    Processes image for CNNs
    :param pil_im : PIL image or numpy array to process
    :param resize_im : Resize to 224 or not
    :return:
        im_as_var (torch variable): Variable that contains processed float tensor
    """

    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print('Could not transform PIL_img to a PIL Image object. Please check input.')

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to C * W * H

    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]

    # Convert to float tensor
    # im_as_ten = torch.from_numpy(im_as_arr).float()
    # # Add one more channel. Tensor shape: 1, 3, 224, 224
    # im_as_ten.unsqueeze_(0)
    # # Convert to Pytorch variable
    # im_as_var = Variable(im_as_ten, requires_grad=True)
    # im_as_var = torch.tensor(im_as_ten)

    im_as_ten = torch.tensor(im_as_arr, dtype=torch.float32, requires_grad=True)
    im_as_ten = torch.unsqueeze(im_as_ten, dim=0)
    return im_as_ten


def get_positive_negative_saliency(gradient):
    """
    Generates positive and negative saliency maps based on the gradient
    :param gradient: Gradient of the operation to visualize, type numpy
    :return: pos_saliency, neg_saliency
    :param gradient:
    :return:
    """

    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / - gradient.min())
    return pos_saliency, neg_saliency


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
    Apply heat map on image
    :param org_im : Original PIL Image
    :param activation : Activation map (grayscale) 0-255, type np shape (1, H, W)
    :param colormap_name : Name of the colormap, type string
    :return:
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation.squeeze())

    # Change alpha channel in colormap
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4   # (H, W, 4)

    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    # Apply heat map on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)

    return no_trans_heatmap, heatmap_on_image


def save_class_activation_images(org_img, activation_map, file_name, colormap='hot'):
    """
    Save cam activation map and activation map on the original image
    :param colormap:
    :param org_img: Original PIL image
    :param activation_map: Activation map (grayscale) 0-255, type np
    :param file_name: File name of the exported image, type string
    :return:
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, colormap)
    # Save colored heatmap
    path_to_file = os.path.join('./results', file_name+'_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on image
    path_to_file = os.path.join('./results', file_name+'_Heatmap_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # Save grayscale heatmap
    path_to_file = os.path.join('./results', file_name + '_Grayscale.png')
    save_image(activation_map, path_to_file)


