import os
import pickle
import numpy as np
import scipy
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from deepface.extendedmodels import Race
from metrics import metric_base


class KID(metric_base.MetricBase):
    def __init__(self, minibatch_per_gpu, use_cached_real_stats=False, **kwargs):
        super().__init__(**kwargs)
        self.minibatch_per_gpu = minibatch_per_gpu
        self.use_cached_real_stats = use_cached_real_stats

        
    def compute_kid(self, feat_real, feat_fake, num_subsets=100, max_subset_size=1000):
        n = feat_real.shape[1]
        m = min(min(feat_real.shape[0], feat_fake.shape[0]), max_subset_size)
        t = 0
        for _subset_idx in range(num_subsets):
            x = feat_fake[np.random.choice(feat_fake.shape[0], m, replace=False)]
            y = feat_real[np.random.choice(feat_real.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
            b = (x @ y.T / n + 1) ** 3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
        return t / num_subsets / m

    
    def _evaluate(self, Gs, G_kwargs, num_gpus, **_kwargs): # pylint: disable=arguments-differ
        minibatch_size = num_gpus * self.minibatch_per_gpu

        with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
            feature_net = pickle.load(f)
        
        race_model = Race.loadModel()
        labels = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]
        n_classes = 6
        nfeat = feature_net.output_shape[1]
        
        feats_all = {}
        for l in labels:
            feats_all[l] = []
        
        
        for images, _labels, num in self._iterate_reals(minibatch_size):

            if images.shape[1] == 1:
                images = np.tile(images, [1, 3, 1, 1])

            race_labels = []
            images_temp = tf.transpose(tf.cast(tf.convert_to_tensor(images), tf.float32), perm=[0, 2, 3, 1])
            images_temp = tf.image.resize(images_temp, (224, 224))
            race_labels = race_model(images_temp)

            race_labels = tflib.run(race_labels)

            race_labels = np.argmax(race_labels, 1)

            feats = feature_net.run(images, num_gpus=num_gpus, assume_frozen=True)
            
            for i in range(len(feats)):
                feats_all[labels[race_labels[i]]].append(feats[i])
        
        for i in range(n_classes):
            for j in range(i, n_classes):
                if i ==j:
                    x = np.array(feats_all[labels[i]])
                    kid = self.compute_kid(x[:len(x)//2], x[len(x)//2+1:])
                else:
                    x = np.array(feats_all[labels[i]])
                    y = np.array(feats_all[labels[j]])
                    kid = kid = self.compute_kid(x, y)
                
            print(f"KID score between {labels[i]} and {labels[j]} is equal to {kid}")
                
        
        
        self._report_result(0)