
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
# from deepface.extendedmodels import Race
# from collections import Counter
import torch 
from niqe.niqe import niqe as niqe_score
from PIL import Image
from pyiqa import create_metric
# from piq import niqe as niqe_
# from tfio.experimental.color import rgb_to_ycbcr
# ----------------------------------------------------------------------------


class brisque(metric_base.MetricBase):
    def __init__(self, num_images, num_splits, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.num_splits = num_splits
        self.minibatch_per_gpu = minibatch_per_gpu
    
#     @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
    def get_niqe_score(self, image):
        return niqe_score(image)
    

    def _evaluate(self, Gs, G_kwargs, num_gpus, **_kwargs):
        minibatch_size = num_gpus * self.minibatch_per_gpu
#         niqe_metric = niqe_()
        metric = create_metric('brisque', test_y_channel=True, device=torch.device("cpu"))
        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                # inception_clone = inception.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)
                images = Gs_clone.get_output_for(latents, labels, **G_kwargs)
#                 images = tf.transpose(images, perm=[0, 2, 3, 1])
#                 images = rgb_to_ycbcr(images)
#                 images = tf.image.resize(images, (112, 112))
                images = tflib.convert_images_to_uint8(images)
#                 images = tf.transpose(images, perm=[0, 3, 1, 2])
#                 images = images.numpy()
                
                
                result_expr.append(images)
                
        scores = []
        for begin in range(0, self.num_images, minibatch_size):
            self._report_progress(begin, self.num_images)
            # end = min(begin + minibatch_size, self.num_images)
            
            out = tflib.run(result_expr)
            for i in range(len(out)):
                out2 = out[i]
                out2 = torch.Tensor(out2)/255.
                quality = metric(out2, None).cpu()
                scores = np.append(scores, quality)

        self._report_result(np.mean(scores), suffix='_mean')
        self._report_result(np.std(scores), suffix='_std')

# ----------------------------------------------------------------------------



