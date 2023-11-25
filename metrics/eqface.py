# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Inception Score (IS)."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
# from deepface.extendedmodels import Race
# from collections import Counter
import torch
import cv2
from models.model_resnet import ResNet, FaceQuality
import os
import argparse
import shutil
import numpy as np

# ----------------------------------------------------------------------------


class EQFace(metric_base.MetricBase):
    def __init__(self, num_images, num_splits, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.num_splits = num_splits
        self.minibatch_per_gpu = minibatch_per_gpu
    
    def load_state_dict(self, model, state_dict):
        all_keys = {k for k in state_dict.keys()}
        for k in all_keys:
            if k.startswith('module.'):
                state_dict[k[7:]] = state_dict.pop(k)
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        if len(pretrained_dict) == len(model_dict):
            print("all params loaded")
        else:
            not_loaded_keys = {k for k in pretrained_dict.keys() if k not in model_dict.keys()}
            print("not loaded keys:", not_loaded_keys)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    def get_face_quality(self, backbone, quality, device, ccropped):
#         resized = cv2.resize(img, (112, 112))
#         ccropped = resized[...,::-1] # BGR to RGB
        # load numpy to tensor
        ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
        ccropped = np.reshape(ccropped, [1, 3, 112, 112])
        ccropped = np.array(ccropped, dtype = np.float32)
        ccropped = (ccropped - 127.5) / 128.0
        ccropped = torch.from_numpy(ccropped)

        # extract features
        backbone.eval() # set to evaluation mode
        with torch.no_grad():
            _, fc = backbone(ccropped.to(device), True)
            s = quality(fc)[0]

        return s.cpu().numpy()


    def _evaluate(self, Gs, G_kwargs, num_gpus, **_kwargs):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        # inception = misc.load_pkl('https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/inception_v3_softmax.pkl')
        # activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)
        backbone = '/scratch/aj3281/FaceQuality/backbone.pth'
        quality = '/scratch/aj3281/FaceQuality/quality.pth'
        BACKBONE = ResNet(num_layers=100, feature_dim=512)
        QUALITY = FaceQuality(512 * 7 * 7)
        
        checkpoint = torch.load(backbone, map_location='cpu')
        self.load_state_dict(BACKBONE, checkpoint)
        
        checkpoint = torch.load(quality, map_location='cpu')
        self.load_state_dict(QUALITY, checkpoint)
        DEVICE = torch.device("cpu")
#         DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        BACKBONE.to(DEVICE)
        QUALITY.to(DEVICE)
        BACKBONE.eval()
        QUALITY.eval()
        print("Loaded models - DONE")
        
        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                # inception_clone = inception.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)
                images = Gs_clone.get_output_for(latents, labels, **G_kwargs)
                images = tf.transpose(images, perm=[0, 2, 3, 1])
                images = tf.image.resize(images, (112, 112))
                images = tflib.convert_images_to_uint8(images)
#                 images = images.numpy()
                
                result_expr.append(images)
                

        # Calculate activations for fakes.
        scores = []
        for begin in range(0, self.num_images, minibatch_size):
            self._report_progress(begin, self.num_images)
            # end = min(begin + minibatch_size, self.num_images)
            out = tflib.run(result_expr)
            for j in range(len(out)):

                out2 = out[j]
                for i in range(len(out2)):

                    quality = self.get_face_quality(BACKBONE, QUALITY, DEVICE, out2[i])
                    scores.append(quality)

        self._report_result(np.mean(scores), suffix='_mean')
        self._report_result(np.std(scores), suffix='_std')

# ----------------------------------------------------------------------------

