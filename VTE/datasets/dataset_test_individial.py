from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.data as data

from PIL import Image


def find_images_and_targets(date_id = None):
    labels = []
    filenames = []
    
    # -----testing patient video frame
    index = os.listdir(date_id)
    index.sort(key=lambda x : int(x.split('.')[0]))
    for pic_train in index:
        label = int(pic_train.split('.')[0])
        labels.append(label)
        filenames.append(date_id + '/' + pic_train)


    images_and_targets = [(f, l) for f, l in zip(filenames, labels)]

    return images_and_targets


class Dataset_individial(data.Dataset):

    def __init__(
            self,
            load_bytes=False,
            transform=None,
            class_map='',
            training = True,
            data_id= None):

        images = find_images_and_targets(date_id = data_id)
        self.samples = images
        self.imgs = self.samples
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

    def filenames(self, basename=False, absolute=False):
        fn = lambda x: x
        if basename:
            fn = os.path.basename
        elif not absolute:
            fn = lambda x: os.path.relpath(x, self.root)
        return [fn(x[0]) for x in self.samples]
