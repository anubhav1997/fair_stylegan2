

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
# from deepface.extendedmodels import Race
from tensorflow2_cifar.models import ResNet
from collections import Counter
from tensorflow.python.keras.models import load_model
import os 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
tf.compat.v1.disable_v2_behavior()

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# ----------------------------------------------------------------------------


class FairnessMetricCifar10(metric_base.MetricBase):
    def __init__(self, num_images, num_splits, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.num_splits = num_splits
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, G_kwargs, num_gpus, **_kwargs):
        minibatch_size = num_gpus * self.minibatch_per_gpu

        race_model = ResNet('resnet18', 10)
#         race_model.load_weights("resnet18_cifar10.pth")
        race_model.load_weights("resnet18_cifar10_255")

        
        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)
                images = Gs_clone.get_output_for(latents, labels, **G_kwargs)
                images = tf.dtypes.cast(images, tf.float32)
                images = tf.transpose(images, perm=[0, 2, 3, 1])

                output = race_model(images, training=False)
                result_expr.append(output)

        # Calculate activations for fakes.
        preds = []
        for begin in range(0, self.num_images, minibatch_size):
            self._report_progress(begin, self.num_images)
            out = tflib.run(result_expr)
            for i in range(len(out)):
                
                out2 = np.argmax(out[i], 1)
                preds = np.append(preds, out2)

        scores = []
        for i in range(self.num_splits):
            outputs = preds[i * self.num_images // self.num_splits: (i + 1) * self.num_images // self.num_splits]
            # counts = dict()
            # for i in outputs:
            #     counts[i] = counts.get(i, 0) + 1
            counts = Counter(outputs)
            score = 0
            for key in counts.keys():
                score += (1. / 10. - counts[key] / len(outputs)) ** 2
            scores.append(np.sqrt(score))

        self._report_result(np.mean(scores), suffix='_mean')
        self._report_result(np.std(scores), suffix='_std')

# ----------------------------------------------------------------------------

