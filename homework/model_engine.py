# TODO(pawelb): general description of this file
# TODO(pawelb): model memory cleanup
# TODO(pawelb): smarter imports: maybe theano should be only loaded when needed?
# TODO(pawelb): better sample demonstration: reuse the same window?
# TODO(pawelb_: enable scripting: e.g. python model train save
from __future__ import division,print_function
import cmd

import os, sys, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

# TODO(pawelb): this line is required to allow importing of modules from ../fastai-course/.../nbs/
# move the utils here? rewrite it?
lib_path = os.path.abspath('../fastai-course/deeplearning1/nbs/')
sys.path.append(lib_path)
# import utils; reload(utils)
#import utils; reload(utils)
from utils import plots

import vgg16; reload(vgg16)
from vgg16 import Vgg16

# TODO(pawelb): for cleanup and direct KERAS access
from keras import backend as K
from keras.models import load_model

# for config file
import ConfigParser
import os.path


CONFIG_PATH = './model.rc'
# Valid config keys
CONFIG_KEYS = ['name', 'model_file', 'data', 'train_batch']
# Config section that will be used for config updates
CONFIG_SECTION = 'model'
# Sample config. If config file does not exist, it will be created with SAMPLE_CONFIG as content
SAMPLE_CONFIG = """
[model]
name=Dogs vs Cats
#model_file=./models/dogs-vs-cats.%Y%m%d%H%M%S
model_file=./models/dogs-vs-cats
data=../fastai-course/deeplearning1/nbs/data/dogscats/sample100/
train_batch=4
"""
MODEL_FILE_SUFFIX = '.h5'
CLASSES_FILE_SUFFIX = '.classes'
# Valid batch names for sample predictions
SAMPLE_BATCHES = ['train', 'valid']
DEFAULT_SAMPLES = 4


class ModelRunner(cmd.Cmd):
    """Simple command processor example."""

    prompt = '\n$> '

    def emptyline(self):
        """ Will cause empty lines do nothing (default is 'repeat last command') """
        return False

    def __init__(self):
        cmd.Cmd.__init__(self)
        self.model = None
        self.batches = {}
        self.config = None
        self.init_config()

    def init_config(self):
        if not os.path.exists(CONFIG_PATH):
            f = open(CONFIG_PATH, 'w')
            f.write(SAMPLE_CONFIG)
            f.close()
        config = ConfigParser.RawConfigParser()
        config.read(CONFIG_PATH)
        self.config = config
        self.print_config()

    def get_model(self):
        if self.model is None:
            self.model = Vgg16()
            self.batches = {}
        return self.model

    def get_batches(self, batch_type, batch_size):
        """ Get batches caches the batches iterators. Useful for viewing multiple sample sets"""
        if batch_type not in self.batches:
            model = self.get_model()
            path = self.config.get(CONFIG_SECTION, 'data')
            batches = model.get_batches(path+batch_type, batch_size=batch_size)
            self.batches[batch_type] = batches
        return self.batches[batch_type]

    def do_train(self, line):
        # TODO(pawelb): save the accuracy info from training
        batch_size = self.config.getint(CONFIG_SECTION, 'train_batch')
        path = self.config.get(CONFIG_SECTION, 'data')
        print('Training the model\nData: {0}\nBatch size = {1}'.format(path, batch_size))
        model = self.get_model()
        # Grab a few images at a time for training and validation.
        # NB: They must be in subdirectories named based on their category
        batches = model.get_batches(path+'train', batch_size=batch_size)
        val_batches = model.get_batches(path+'valid', batch_size=batch_size)
        # NOTE(pawelb): remove this line, if you want general Vgg16 model (with ImageNet classes).
        # Finetune is required to remap Vgg16 classes to our-defined classes.
        model.finetune(batches)
        model.fit(batches, val_batches, nb_epoch=1)

    def do_save(self, model_path_override):
        # TODO(pawelb): implement some smart options for overriding
        path = self.config.get(CONFIG_SECTION, 'model_file')
        if model_path_override:
            path = model_path_override
        print('Saving model and classes to files: {0}.h5 and {1}.classes'.format(path, path))
        # Create the directory, if necessary
        dir = os.path.dirname(os.path.abspath(path))
        if not os.path.exists(dir):
            os.makedirs(dir, 0755)
        # Save the model file and classes file
        vgg_model = self.get_model()
        vgg_model.model.save(path + MODEL_FILE_SUFFIX)
        classes_file = open(path + CLASSES_FILE_SUFFIX, 'w')
        classes_file.write('\n'.join(vgg_model.classes))
        classes_file.close()

    def do_clear(self, line):
        print('clears current model from memory (should release GPU memoery)')
        del self.model
        self.model = None
        #K.clear_session()
        #load_model()

    def do_load(self, model_path_override):
        # TODO(pawelb): implement some smart options for overriding the in-memory model
        path = self.config.get(CONFIG_SECTION, 'model_file')
        if model_path_override:
            path = model_path_override

        print('Loading model from files: {0}.h5 and {1}.classes'.format(path, path))
        loaded_model = load_model(path + MODEL_FILE_SUFFIX)
        classes = open(path + CLASSES_FILE_SUFFIX).read().splitlines()

        vgg_model = Vgg16()
        vgg_model.model = loaded_model
        vgg_model.classes = classes
        self.model = vgg_model

    def do_sample(self, line):
        sample = self.parse_sample(line)
        if sample[1] is None:
            return False

        model = self.get_model()
        batches = self.get_batches(sample[0], sample[1])

        imgs, labels = next(batches)
        plt.ion()
        plots(imgs, titles=labels)
        predictions = model.predict(imgs, True)
        print(predictions)

    def parse_sample(self, line):
        batch = [SAMPLE_BATCHES[0], DEFAULT_SAMPLES]
        if not line:
            return batch

        if line:
            tokens = line.split()
            if len(tokens) > 2:
                print('Invalid syntax. Expected: sample <batch-type> <optional-batch-size>')
                return [None, None]
            batch_type = tokens[0]
            if batch_type not in SAMPLE_BATCHES:
                print('Invalid batch name: {0}. Should be one of: {1}'.format(batch_type, SAMPLE_BATCHES))
                return [None, None]
            else:
                batch[0] = batch_type

            if len(tokens) == 2:
                # check if we have integer
                try:
                    batch_size = int(tokens[1])
                    batch[1] = batch_size
                except:
                    print('Invalid batch size: {0}. Expected integer.'.format(tokens[1]))
                    return [None, None]
            return batch

    def do_config(self, value):
        if not value:
            self.print_config()
        else:
            self.update_config(value)

    def print_config(self):
        for section in self.config.sections():
            print('[{0}]'.format(section))
            for option in self.config.options(section):
                s = '{0}={1}'.format(option, self.config.get(section, option))
                print(s)

    def update_config(self, value):
        # Update the config
        tokens = value.split('=')
        if len(tokens) != 2:
            print('Invalid config syntax. Expected key=value')
            return False
        key, value = tokens[0], tokens[1]
        if key not in CONFIG_KEYS:
            print('Invalid config key. Valid keys: ' + CONFIG_KEYS)
        self.config.set(CONFIG_SECTION, key, value)
        # Write new config to file and reload it
        f = open(CONFIG_PATH, 'w')
        self.config.write(f)
        f.close()
        self.config = None
        self.init_config()

    def do_kaggle(self, line):
        #path = 'data/dogs-vs-cats-kernels-edition/test10/'
        path = 'data/dogs-vs-cats-kernels-edition/test/'
        batch_size = 8

        # count files in unknown/*.jpg
        total_files = len(glob(path + 'unknown/*.jpg'))

        #x = self.get_model().get_batches(path+'unknown', shuffle=False, batch_size=batch_size)
        model = self.get_model()
        batch = model.get_batches(path, shuffle=False, batch_size=batch_size)
        filenames = batch.filenames[:]
        predictions = []

        while True:
            try:
                imgs, unused_labels = next(batch)
            except StopIteration:
                break
            frag_predictions = model.predict(imgs, True)
            for i in xrange(len(imgs)):
                fname = filenames[0]
                filenames = filenames[1:]
                confidence = frag_predictions[0][i]
                pred_class = frag_predictions[1][i]
                pred_class_name = frag_predictions[2][i]
                predictions.append([fname, confidence, pred_class, pred_class_name])
            # batch is an infinite iterator; we need to track number of already
            # processed images by ourselves.
            print(len(predictions))
            if len(predictions) == total_files:
                break

        # for p in predictions:
        #     print(p)
        pred_csv = open('predictions.csv', 'w')
        pred_csv.write('id,label\n')
        for p in predictions:
            label = p[0].split('/')[1].split('.')[0]
            id = p[2]
            pred_csv.write('{0},{1}\n'.format(label, id))
        pred_csv.close()

    def do_exit(self, line):
        return True

if __name__ == '__main__':
    r = ModelRunner()
    r.cmdloop()
