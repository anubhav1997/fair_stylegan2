

import os
import pickle
import numpy as np
import scipy
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from deepface.extendedmodels import Race
from metrics import metric_base
# tf.enable_eager_execution()
#----------------------------------------------------------------------------

class FID(metric_base.MetricBase):
    def __init__(self, minibatch_per_gpu, use_cached_real_stats=False, **kwargs):
        super().__init__(**kwargs)
#         self.max_reals = max_reals
#         self.num_fakes = num_fakes
        self.minibatch_per_gpu = minibatch_per_gpu
        self.use_cached_real_stats = use_cached_real_stats

        
    def fid_score(self, mu_real, sigma_real, mu_fake, sigma_fake):
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)  # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2 * s)
        # self._report_result(np.real(dist))
        return np.real(dist)

    
    def _evaluate(self, Gs, G_kwargs, num_gpus, **_kwargs): # pylint: disable=arguments-differ
        minibatch_size = num_gpus * self.minibatch_per_gpu
#        with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/inception_v3_features.pkl') as f:

        with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
            feature_net = pickle.load(f)
        
        race_model = Race.loadModel()
        labels = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]
        n_classes = 6
        nfeat = feature_net.output_shape[1]
        mu_real = np.zeros((n_classes, nfeat))
        mu_real2 = np.zeros((n_classes, nfeat))
#         y=5
        sigma_real = np.zeros([n_classes, nfeat, nfeat])
        sigma_real2 = np.zeros([n_classes, nfeat, nfeat])
        
        num_real = np.zeros(n_classes)
        num_real2 = np.zeros(n_classes)
        
        for images, _labels, num in self._iterate_reals(minibatch_size):

            if images.shape[1] == 1:
                images = np.tile(images, [1, 3, 1, 1])

            race_labels = []
            images_temp = tf.transpose(tf.cast(tf.convert_to_tensor(images), tf.float32), perm=[0, 2, 3, 1])
            images_temp = tf.image.resize(images_temp, (224, 224))
            race_labels = race_model(images_temp)

            race_labels = tflib.run(race_labels)

            race_labels = np.argmax(race_labels, 1)


            i = 0 
            for feat in list(feature_net.run(images, num_gpus=num_gpus, assume_frozen=True))[:num]:
                if np.random.choice(a=[False, True]):    
                    mu_real[race_labels[i]] += feat
                    sigma_real[race_labels[i]] += np.outer(feat, feat)
                    num_real[race_labels[i]] += 1
                else:
                    mu_real2[race_labels[i]] += feat
                    sigma_real2[race_labels[i]] += np.outer(feat, feat)
                    num_real2[race_labels[i]] += 1
                
                i += 1

        

        for i in range(n_classes):
                           
            mu_real[i] /= num_real[i]
            sigma_real[i] /= num_real[i]
            sigma_real[i] -= np.outer(mu_real[i], mu_real[i])
            
            mu_real2[i] /= num_real2[i]
            sigma_real2[i] /= num_real2[i]
            sigma_real2[i] -= np.outer(mu_real2[i], mu_real2[i])
        
        
        for i in range(n_classes):
#             for j in range(i, n_classes):
                print(num_real[i], num_real2[i])

                fid = self.fid_score(mu_real[i], sigma_real[i], mu_real2[i], sigma_real2[i])
                print(f"FID score between {labels[i]} and {labels[i]} is equal to {fid}")
                
        
        
        self._report_result(0)