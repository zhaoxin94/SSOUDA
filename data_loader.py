from torch.utils.data import Dataset
import csv
import os
from PIL import Image
import numpy as np

dataset_dict = {
    'UCMD': {
        "source": {
            "train_path": 'dataset/source_index/UCMD/train.csv',
            "val_path": 'dataset/source_index/UCMD/val.csv',
            "test_path": 'dataset/source_index/UCMD/test.csv',
            'database_path': 'dataset/source_index/UCMD/database.csv'
        },
        "target": {
            "train_path":
            'dataset/target_index/UCMD/train.csv',
            'class_names': [
                'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike',
                'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles',
                'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains',
                'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan',
                'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork',
                'unknown'
            ]
        }
    },
    'NWPU': {
        "source": {
            "train_path": 'dataset/source_index/NWPU/train.csv'
        },
        "target": {
            "train_path":
            'dataset/target_index/NWPU/train.csv',
            "val_path":
            'dataset/target_index/NWPU/val.csv',
            "test_path":
            'dataset/target_index/NWPU/test.csv',
            'database_path':
            'dataset/target_index/NWPU/database.csv',
            'class_names': [
                'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike',
                'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles',
                'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains',
                'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan',
                'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork',
                'unknown'
            ]
        }
    },
    'AID': {
        "source": {
            "train_path": 'dataset/source_index/AID/train.csv'
        },
        "target": {
            "train_path":
            'dataset/target_index/AID/train.csv',
            'class_names': [
                'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike',
                'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles',
                'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains',
                'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan',
                'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork',
                'unknown'
            ]
        }
    }
}

rs_dataset_name = ['UCMD', 'NWPU']


def get_rs_class_name(dataset_name):
    class_names = dataset_dict[dataset_name]['target']['class_names']
    return class_names


class RS_dataset(Dataset):
    def __init__(self, root, index_file, transform, strong_transform=None):
        with open(index_file, 'r') as fh:
            reader = csv.reader(fh)
            img_paths = []
            labels = []
            for line in reader:
                img_path = os.path.join(root, line[0])
                img_paths.append(img_path)
                labels.append(int(line[1]))

        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_paths)


class RS_dataset_imgpath(Dataset):
    def __init__(self, root, index_file, transform):

        with open(index_file, 'r') as fh:
            reader = csv.reader(fh)
            img_paths = []
            labels = []
            for line in reader:
                img_path = os.path.join(root, line[0])
                img_paths.append(img_path)
                labels.append(int(line[1]))

        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_path

    def __len__(self):
        return len(self.img_paths)


def get_rs_dataset(root, source, dataset_name, transform, appli):

    if source:
        if appli == 'train':
            data_csv = dataset_dict[dataset_name]['source']['train_path']
        elif appli == 'val':
            data_csv = dataset_dict[dataset_name]['source']['val_path']
        elif appli == 'test':
            data_csv = dataset_dict[dataset_name]['source']['test_path']
        else:
            data_csv = dataset_dict[dataset_name]['source']['database_path']
    else:
        if appli == 'train':
            data_csv = dataset_dict[dataset_name]['target']['train_path']
        elif appli == 'val':
            data_csv = dataset_dict[dataset_name]['target']['val_path']
        elif appli == 'test':
            data_csv = dataset_dict[dataset_name]['target']['test_path']
        else:
            data_csv = dataset_dict[dataset_name]['target']['database_path']

    dataset = RS_dataset(root, index_file=data_csv, transform=transform)

    return dataset


def get_rs_dataset_imgpath(root,
                           dataset_name,
                           transform,
                           appli,
                           source='target'):

    if appli == 'train':
        dataset_csv = dataset_dict[dataset_name][source]['train_path']
    elif appli == 'val':
        dataset_csv = dataset_dict[dataset_name][source]['val_path']
    elif appli == 'test':
        dataset_csv = dataset_dict[dataset_name][source]['test_path']
    else:
        dataset_csv = dataset_dict[dataset_name][source]['database_path']

    dataset = RS_dataset_imgpath(root,
                                 index_file=dataset_csv,
                                 transform=transform)

    return dataset
