import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from deepface.extendedmodels import Race
from collections import Counter


# ----------------------------------------------------------------------------


class Counts(metric_base.MetricBase):
    def __init__(self, num_images, num_splits, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.num_splits = num_splits
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, G_kwargs, num_gpus, **_kwargs):
        minibatch_size = num_gpus * self.minibatch_per_gpu
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

        # Calculate activations for fakes.
        preds = []
        for begin in range(0, self.num_images, minibatch_size):
            self._report_progress(begin, self.num_images)
            # end = min(begin + minibatch_size, self.num_images)
            out = tflib.run(result_expr)
            for i in range(len(out)):
                
                out2 = np.argmax(out[i], 1)
                preds = np.append(preds, out2)
                
        counts = Counter(preds)
        print(counts)
        self._report_result(0.0)



# ----------------------------------------------------------------------------

