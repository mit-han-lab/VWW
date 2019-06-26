import torch
import torch.utils.data as data
import os
import json
from PIL import Image

__all__ = ['VWWDataset']


class VWWDataset(data.Dataset):
    def __init__(self, dataset_dir='/dataset/coco/', anno_dir='/home/jilin/workspace/visual_wake_words/torch',
                 split='train', transform=None):
        assert split in ['train', 'val', 'minival']
        if anno_dir is None:  # since now we cannot modify /dataset path, we store the labels in home.
            anno_dir = dataset_dir
        self.dataset_dir = dataset_dir
        self.anno_dir = anno_dir
        self.split = split

        anno_path = os.path.join(anno_dir, 'instances_visualwakewords_{}2014.json'.format(split))
        with open(anno_path) as f:
            labels = json.load(f)  # keys: 'images', 'annotations', 'categories'

        file_names = [l['file_name'] for l in labels['images']]

        self.data_pairs = []
        for f in file_names:
            image_id = f.split('.')[0].split('_')[-1]
            image_id = str(int(image_id))
            this_l = labels['annotations'][image_id]
            assert len(this_l) == 1
            this_l = this_l[0]['label']
            self.data_pairs.append((f, this_l))

        self.transform = transform

    def __getitem__(self, index):
        f_name, label = self.data_pairs[index]

        img = Image.open(os.path.join(self.dataset_dir, '{}2014'.format(self.split.replace('mini', '')),
                                      f_name)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_pairs)
