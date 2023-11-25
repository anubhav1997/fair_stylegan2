




import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import dnnlib#.tflib as tflib
import dnnlib.tflib 


# from metrics import metric_base
# from deepface.extendedmodels import Race
from tensorflow2_cifar.models import ResNet
from collections import Counter
from tensorflow.python.keras.models import load_model
import os 
import pickle 

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# tf.compat.v1.disable_v2_behavior()

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)



# ----------------------------------------------------------------------------


# class FairnessMetricCifar10(metric_base.MetricBase):
#     def __init__(self, num_images, num_splits, minibatch_per_gpu, **kwargs):
#         super().__init__(**kwargs)
#         self.num_images = num_images
#         self.num_splits = num_splits
#         self.minibatch_per_gpu = minibatch_per_gpu

#     def _evaluate(self, Gs, G_kwargs, num_gpus, **_kwargs):





# minibatch_size = num_gpus * minibatch_per_gpu
# inception = misc.load_pkl('https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/inception_v3_softmax.pkl')
# activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)
#         race_model = Race.loadModel()

# dnnlib.tflib.init_tf()
minibatch_size = 128 

network_pkl = "./training-runs/00076-cifar10_imbalance_tfrecord-mirror-stylegan2-noaug/network-snapshot-025000.pkl"

with dnnlib.util.open_url(network_pkl) as f:
    _G, _D, Gs = pickle.load(f)
    Gs.print_layers()

        
race_model = ResNet('resnet18', 10)
ckpt_path = './tensorflow2_cifar/checkpoints/{:s}/'.format("resnet18")
ckpt = tf.train.Checkpoint(model=race_model)
manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)

# Load checkpoint
print('==> Resuming from checkpoint...')
assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'
ckpt.restore(manager.latest_checkpoint)


#         "/scratch/aj3281/stylegan2-ada/tensorflow2_cifar/checkpoints/resnet18"
#         race_model.load_weights("/scratch/aj3281/stylegan2-ada/tensorflow2_cifar/resnet18_cifar10")
#         race_model.load_weights("/scratch/aj3281/stylegan2-ada/tensorflow2_cifar/checkpoints/resnet18/ckpt-53")
#         ckpt_path = tf.train.latest_checkpoint(race_model_path)
#         ckpt.restore(ckpt_path)

#         image = tf.zeros([5, 32, 32, 3], tf.float32)
#         out = race_model(image)
#         out.eval(session=tf.compat.v1.Session()) 
#         print(out)


# Construct TensorFlow graph.
result_expr = []
preds = []
for begin in range(0, self.num_images, minibatch_size):

        latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
        labels = self._get_random_labels_tf(self.minibatch_per_gpu)
        images = Gs_clone.get_output_for(latents, labels, **G_kwargs)
        images = tf.dtypes.cast(images, tf.float32)
        images = tf.transpose(images, perm=[0, 2, 3, 1])
        print(images.shape)
        output = race_model(images, training=False)
        out2 = np.argmax(output, 1)
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

print(np.mean(scores), suffix='_mean')
print(np.std(scores), suffix='_std')
