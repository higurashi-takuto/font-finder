import glob
import random
import argparse

import numpy as np
import chainer
from chainer import cuda
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
import chainer.datasets.image_dataset


# define network
class Network(chainer.Chain):

    def __init__(self, n_class):
        super(Network, self).__init__(
            conv1=L.Convolution2D(None, 64, 3, pad=1),
            conv2=L.Convolution2D(None, 128, 3, pad=1),
            conv3=L.Convolution2D(None, 256, 1),
            conv4=L.Convolution2D(None, 512, 1),
            conv5=L.Convolution2D(None, 1024, 1),
            conv6=L.Convolution2D(None, n_class, 1),

            norm1=L.BatchNormalization(64),
            norm2=L.BatchNormalization(128),
            norm3=L.BatchNormalization(256),
            norm4=L.BatchNormalization(512),
            norm5=L.BatchNormalization(1024),
        )

    def __call__(self, x, test=False):
        h1 = F.relu(F.max_pooling_2d(self.norm1(self.conv1(x), test=test), 2))
        h2 = F.relu(F.max_pooling_2d(self.norm2(self.conv2(h1), test=test), 2))
        h3 = F.relu(self.norm3(self.conv3(h2), test=test))
        h4 = F.relu(self.norm4(self.conv4(h3), test=test))
        h5 = F.relu(self.norm5(self.conv5(h4), test=test))
        h6 = self.conv6(h5)
        y = F.average_pooling_2d(h6, (h6.shape[2], h6.shape[3]))
        y = y[:, :, 0, 0]
        return y


parser = argparse.ArgumentParser(description='Face classifier')
parser.add_argument('--directory', '-d', default='data',
                    help='Directory of images')
parser.add_argument('--batchsize', '-b', type=int, default=20,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=200,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
args = parser.parse_args()

xp = cuda.cupy if args.gpu >= 0 else np

# set directory
font_dirs = glob.glob('{}/*'.format(args.directory).replace('//', '/'))
n_class = len(font_dirs)
f = open('list.txt', 'w')
for x in font_dirs:
    f.write(str(x) + "\n")
f.close()

# set list
data = []

# load data
for i, font_dir in enumerate(font_dirs):
    for font in glob.glob('{}/*'.format(font_dir)):
        data.append([font, i])

# data is list of pairs([image, label])

# devide data for train and test
print(len(data))
random.shuffle(data)
train = chainer.datasets.LabeledImageDataset(data[len(data) // 10:])
test = chainer.datasets.LabeledImageDataset(data[:len(data) // 10])

# set up iterators
train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                             repeat=False, shuffle=False)

# set up model
model = L.Classifier(Network(n_class))
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    model.to_gpu()

# set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# set up updater
updater = chainer.training.StandardUpdater(train_iter, optimizer,
                                           device=args.gpu)

# set up trainer
trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), 'result')

# add extentions
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.epoch}',))
trainer.extend(extensions.snapshot_object(
    model.predictor, 'model_{.updater.epoch}'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PlotReport(
    ['main/loss', 'validation/main/loss'],
    'epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(
    ['main/accuracy', 'validation/main/accuracy'],
    'epoch', file_name='accuracy.png'))
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.ProgressBar())

trainer.run()
