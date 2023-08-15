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
from deepface.extendedmodels import Race
from collections import Counter


# ----------------------------------------------------------------------------


class FairnessMetric(metric_base.MetricBase):
    def __init__(self, num_images, num_splits, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.num_splits = num_splits
        self.minibatch_per_gpu = minibatch_per_gpu

    # def get_output_for(self, *in_expr: TfExpression, return_as_list: bool = False, **dynamic_kwargs) -> Union[TfExpression, List[TfExpression]]:
    #     """Construct TensorFlow expression(s) for the output(s) of this network, given the input expression(s)."""
    #     assert len(in_expr) == self.num_inputs
    #     assert not all(expr is None for expr in in_expr)
    #
    #     # Finalize build func kwargs.
    #     build_kwargs = dict(self.static_kwargs)
    #     build_kwargs.update(dynamic_kwargs)
    #     build_kwargs["is_template_graph"] = False
    #     build_kwargs["components"] = self.components
    #
    #     # Build TensorFlow graph to evaluate the network.
    #     with tfutil.absolute_variable_scope(self.scope, reuse=True), tf.name_scope(self.name):
    #         assert tf.get_variable_scope().name == self.scope
    #         valid_inputs = [expr for expr in in_expr if expr is not None]
    #         final_inputs = []
    #         for expr, name, shape in zip(in_expr, self.input_names, self.input_shapes):
    #             if expr is not None:
    #                 expr = tf.identity(expr, name=name)
    #             else:
    #                 expr = tf.zeros([tf.shape(valid_inputs[0])[0]] + shape[1:], name=name)
    #             final_inputs.append(expr)
    #         out_expr = self._build_func(*final_inputs, **build_kwargs)
    #
    #     # Propagate input shapes back to the user-specified expressions.
    #     for expr, final in zip(in_expr, final_inputs):
    #         if isinstance(expr, tf.Tensor):
    #             expr.set_shape(final.shape)
    #
    #     # Express outputs in the desired format.
    #     assert tfutil.is_tf_expression(out_expr) or isinstance(out_expr, tuple)
    #     if return_as_list:
    #         out_expr = [out_expr] if tfutil.is_tf_expression(out_expr) else list(out_expr)
    #     return out_expr

    def _evaluate(self, Gs, G_kwargs, num_gpus, **_kwargs):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        # inception = misc.load_pkl('https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/inception_v3_softmax.pkl')
        # activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)
        race_model = Race.loadModel()
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
                images = tf.image.resize(images, (224, 224))
                # images = tflib.convert_images_to_uint8(images)
                result_expr.append(race_model(images))
                # result_expr.append(tf.math.argmax(race_model(images), 0))

        # Calculate activations for fakes.
        preds = []
        for begin in range(0, self.num_images, minibatch_size):
            self._report_progress(begin, self.num_images)
            # end = min(begin + minibatch_size, self.num_images)
            out = tflib.run(result_expr)

            out = np.argmax(out[0], 1)
            preds = np.append(preds, out)
            # activations[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end-begin]

        # print(preds)
        scores = []
        for i in range(self.num_splits):
            outputs = preds[i * self.num_images // self.num_splits: (i + 1) * self.num_images // self.num_splits]
            # counts = dict()
            # for i in outputs:
            #     counts[i] = counts.get(i, 0) + 1
            counts = Counter(outputs)
            score = 0
            for key in counts.keys():
                score += (1. / 6. - counts[key] / len(outputs)) ** 2
            scores.append(np.sqrt(score))

        # Calculate IS.
        # scores = []
        # for i in range(self.num_splits):
        #     part = activations[i * self.num_images // self.num_splits : (i + 1) * self.num_images // self.num_splits]
        #     kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        #     kl = np.mean(np.sum(kl, 1))
        #     scores.append(np.exp(kl))
        self._report_result(np.mean(scores), suffix='_mean')
        self._report_result(np.std(scores), suffix='_std')

# ----------------------------------------------------------------------------
