import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        print(len(self.examples))
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
      
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        # if self.split == 'train':
        #     a = example['report'].split(". ")
        #     random.shuffle(a)
        #     example['report'] = '. '.join(a)
        #     example['report'] = example['report'] + "."
        #     example['report'] = example['report'].replace('.. ', '. ').replace('..', '.')

        example['ids'] = self.tokenizer(example['report'])[:self.max_seq_length]
        example['mask'] = [1] * len(example['ids'])
        report_ids = example['ids']
        report_masks = example['mask']

        # if self.split == "train":
        #     text = example['ids']
        #     if len(text) != 1:
        #         new_text = []
        #         for word in text:
        #             r = random.uniform(0, 1)
        #             if r > 0.2 or word in [0, 1]:
        #                 new_text.append(word)
        #         if len(new_text) == 0:
        #             new_text.append(0)
        #     report_ids = new_text
        #     report_masks = [1] * len(report_ids)

        image = torch.stack((image_1, image_2), 0)
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
