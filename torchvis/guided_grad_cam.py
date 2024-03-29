#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Aug 02 07:50:18 2022

@author: Nacriema

Refs:

"""
import torch

from torchvis.base_cam import BaseCAM
from .activations_and_gradients import ActivationsAndGradients
import numpy as np
from PIL import Image
from torchvis.utils.image import convert_to_grayscale


class GuidedGradCAM(BaseCAM):
    def __init__(self, model, layers):
        super(GuidedGradCAM, self).__init__(model)
        self.activations_and_grads = ActivationsAndGradients(self.model,
                                                             [self._get_module_by_name(layer) for layer in layers])

    def generate_gradients(self, input_image, target_class):
        # Step 1
        model_output = self.activations_and_grads(input_image)

        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())

        # Step 2, 3. Prepare the tensor for the backprop pass
        tensor_backprop = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        tensor_backprop[0, target_class] = 1

        # Step 4. Perform backpropagation
        self.model.zero_grad()

        model_output.backward(gradient=tensor_backprop, retain_graph=True)

        # Step 5.
        # Get the output feature as well as the guided gradient at that CNN layer
        conv_output = torch.squeeze(self.activations_and_grads.activations[1]).data.numpy()
        guided_gradient = torch.squeeze(self.activations_and_grads.gradients[1]).data.numpy()

        vanilla_grads = torch.squeeze(self.activations_and_grads.gradients[0]).data.numpy()
        grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)

        weights = np.mean(guided_gradient, axis=(1, 2))
        weights = weights[:, np.newaxis, np.newaxis]
        grad_cam = np.multiply(conv_output, weights)
        grad_cam = np.sum(grad_cam, axis=0)

        # Step 7. Apply ReLU into the grad cam
        grad_cam = np.clip(grad_cam, 0, None)

        # Step 8. Normalize, greater than 0 and range 0-1
        grad_cam = np.maximum(grad_cam, 0)
        grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam))
        grad_cam = np.uint8(grad_cam * 255)

        # Step 9. Upscale to fit the reference image
        grad_cam = np.uint8(
            Image.fromarray(grad_cam).resize((input_image.shape[2], input_image.shape[3]), Image.BILINEAR)
        ) / 255.

        # Step 10. Multiply element-wise grad-cam with the gray scale vanilla grads
        guided_grad_cam = np.multiply(grad_cam, grayscale_vanilla_grads)
        return guided_grad_cam

