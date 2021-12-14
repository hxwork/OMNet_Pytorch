import os
import pickle
import random
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.transformations import fetch_transform

_logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    rand_seed = random.randint(0, 2**32 - 1)
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


class ModelNetNpy(Dataset):
    def __init__(self, dataset_path: str, dataset_mode: str, subset: str = "train", categories=None, transform=None):
        """ModelNet40 TS data.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path
        self._subset = subset

        metadata_fpath = os.path.join(self._root, "modelnet_{}_{}.pickle".format(dataset_mode, subset))
        self._logger.info("Loading data from {} for {}".format(metadata_fpath, subset))

        if not os.path.exists(os.path.join(dataset_path)):
            assert FileNotFoundError("Not found dataset_path: {}".format(dataset_path))

        with open(os.path.join(dataset_path, "shape_names.txt")) as fid:
            self._classes = [l.strip() for l in fid]
            self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
            self._idx2category = self._classes

        if categories is not None:
            categories_idx = [self._category2idx[c] for c in categories]
            self._logger.info("Categories used: {}.".format(categories_idx))
            self._classes = categories
        else:
            categories_idx = None
            self._logger.info("Using all categories.")

        self._data = self._read_pickle_files(os.path.join(dataset_path, "modelnet_{}_{}.pickle".format(dataset_mode, subset)),
                                             categories_idx)

        self._transform = transform
        self._logger.info("Loaded {} {} instances.".format(len(self._data), subset))

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def _read_pickle_files(fnames, categories):

        all_data_dict = []
        with open(fnames, "rb") as f:
            data = pickle.load(f)

        for category in categories:
            all_data_dict.extend(data[category])

        return all_data_dict

    def to_category(self, i):
        return self._idx2category[i]

    def __getitem__(self, item):

        data_path = self._data[item]

        # load and process data
        points = np.load(data_path)
        idx = np.array(int(os.path.splitext(os.path.basename(data_path))[0].split("_")[1]))
        label = np.array(int(os.path.splitext(os.path.basename(data_path))[0].split("_")[3]))
        sample = {"points": points, "label": label, "idx": idx}

        if self._transform:
            sample = self._transform(sample)
        return sample

    def __len__(self):
        return len(self._data)


def fetch_dataloader(params):
    _logger.info("Dataset type: {}, transform type: {}".format(params.dataset_type, params.transform_type))
    train_transforms, test_transforms = fetch_transform(params)
    if params.dataset_type == "modelnet_os":
        dataset_path = "./dataset/data/modelnet_os"
        train_categories = [line.rstrip("\n") for line in open("./dataset/data/modelnet40_half1_rm_rotate.txt")]
        val_categories = [line.rstrip("\n") for line in open("./dataset/data/modelnet40_half1_rm_rotate.txt")]
        test_categories = [line.rstrip("\n") for line in open("./dataset/data/modelnet40_half2_rm_rotate.txt")]
        train_categories.sort()
        val_categories.sort()
        test_categories.sort()
        train_ds = ModelNetNpy(dataset_path, dataset_mode="os", subset="train", categories=train_categories, transform=train_transforms)
        val_ds = ModelNetNpy(dataset_path, dataset_mode="os", subset="val", categories=val_categories, transform=test_transforms)
        test_ds = ModelNetNpy(dataset_path, dataset_mode="os", subset="test", categories=test_categories, transform=test_transforms)

    elif params.dataset_type == "modelnet_ts":
        dataset_path = "./dataset/data/modelnet_ts"
        train_categories = [line.rstrip("\n") for line in open("./dataset/data/modelnet40_half1_rm_rotate.txt")]
        val_categories = [line.rstrip("\n") for line in open("./dataset/data/modelnet40_half1_rm_rotate.txt")]
        test_categories = [line.rstrip("\n") for line in open("./dataset/data/modelnet40_half2_rm_rotate.txt")]
        train_categories.sort()
        val_categories.sort()
        test_categories.sort()
        train_ds = ModelNetNpy(dataset_path, dataset_mode="ts", subset="train", categories=train_categories, transform=train_transforms)
        val_ds = ModelNetNpy(dataset_path, dataset_mode="ts", subset="val", categories=val_categories, transform=test_transforms)
        test_ds = ModelNetNpy(dataset_path, dataset_mode="ts", subset="test", categories=test_categories, transform=test_transforms)

    else:
        raise NotImplementedError

    dataloaders = {}
    params.prefetch_factor = 5
    # add defalt train data loader
    train_dl = DataLoader(train_ds,
                          batch_size=params.train_batch_size,
                          shuffle=True,
                          num_workers=params.num_workers,
                          pin_memory=params.cuda,
                          drop_last=True,
                          prefetch_factor=params.prefetch_factor,
                          worker_init_fn=worker_init_fn)
    dataloaders["train"] = train_dl

    # chosse val or test data loader for evaluate
    for split in ["val", "test"]:
        if split in params.eval_type:
            if split == "val":
                dl = DataLoader(val_ds,
                                batch_size=params.eval_batch_size,
                                shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda,
                                prefetch_factor=params.prefetch_factor)
            elif split == "test":
                dl = DataLoader(test_ds,
                                batch_size=params.eval_batch_size,
                                shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda,
                                prefetch_factor=params.prefetch_factor)
            else:
                raise ValueError("Unknown eval_type in params, should in [val, test]")
            dataloaders[split] = dl
        else:
            dataloaders[split] = None

    return dataloaders
