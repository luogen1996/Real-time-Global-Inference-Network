from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as F
import pandas as pd
from utils import DataWrap
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import re
import PIL
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import pickle
import ast
import logging
from torchvision import transforms
import spacy
import cv2
from extended_config import cfg as conf



nlp = spacy.load('en_core_web_md')

def generate_iou_groundtruth(grid_shapes,true_anchor,true_wh):
    """
    :param grid_shapes:   widths and heights for generation (w,h)
    :param true_anchor:  anchor's x and y (x,y)
    :param true_wh:  anchor's width and height (w,h) use for calculate iou
    :return: general iou distribution without any hyperparameter for attention loss
    """
    def cal_single_iou(box1, box2):
        smooth = 1e-7
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max((yi2 - yi1), 0.) * max((xi2 - xi1), 0.)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        iou = (inter_area + smooth) / (union_area + smooth)
        return iou
    IMAGE_WIDTH = grid_shapes[0]
    IMAGE_HEIGHT = grid_shapes[1]


    t_w,t_h=true_wh
    t_x,t_y=true_anchor

    gt_box=[t_x-t_w/2,t_y-t_h/2,t_x+t_w/2,t_y+t_h/2]

    iou_map=np.zeros([IMAGE_WIDTH,IMAGE_HEIGHT])
    for i in range(IMAGE_WIDTH):
        for j in range(IMAGE_HEIGHT):
            iou_map[i,j]=cal_single_iou(gt_box,[max(i-t_w/2,0.),max(j-t_h/2,0.),min(i+t_w/2,IMAGE_WIDTH),min(j+t_h/2,IMAGE_HEIGHT)])

    return iou_map
def pil2tensor(image, dtype: np.dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.expand_dims(a, 2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False))


class NewDistributedSampler(DistributedSampler):
    """
    Same as default distributed sampler of pytorch
    Just has another argument for shuffle
    Allows distributed in validation/testing as well
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)


class ImgQuDataset(Dataset):
    """
    Any Grounding dataset.
    Args:
        train_file (string): CSV file with annotations
        The format should be: img_file, bbox, queries
        Can have same img_file on multiple lines
    """

    def __init__(self, cfg, csv_file, ds_name, split_type='train'):
        self.cfg = cfg
        self.ann_file = csv_file
        self.ds_name = ds_name
        self.split_type = split_type

        # self.image_data = pd.read_csv(csv_file)
        self.image_data = self._read_annotations(csv_file)
        # self.image_data = self.image_data.iloc[:200]
        self.img_dir = Path(self.cfg.ds_info[self.ds_name]['img_dir'])
        self.phrase_len = 50
        self.item_getter = getattr(self, 'simple_item_getter')
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        # std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        return self.item_getter(idx)

    def simple_item_getter(self, idx):
        img_file, annot, q_chosen = self.load_annotations(idx)
        img = PIL.Image.open(img_file).convert('RGB')
        img_ = np.array(img)
        h, w = img.height, img.width

        q_chosen = q_chosen.strip()
        qtmp = nlp(str(q_chosen))
        if len(qtmp) == 0:
            # logger.error('Empty string provided')
            raise NotImplementedError
        qlen = len(qtmp)
        q_chosen = q_chosen + ' PD'*(self.phrase_len - qlen)
        q_chosen_emb = nlp(q_chosen)
        if not len(q_chosen_emb) == self.phrase_len:
            q_chosen_emb = q_chosen_emb[:self.phrase_len]

        q_chosen_emb_vecs = np.array([q.vector for q in q_chosen_emb])
        # qlen = len(q_chosen_emb_vecs)
        # Annot is in x1y1x2y2 format
        target = np.array(annot)
        # img = self.resize_fixed_transform(img)
        img = img.resize((self.cfg.resize_img[0], self.cfg.resize_img[1]))
        # Now target is in y1x1y2x2 format which is required by the model
        # The above is because the anchor format is created
        # in row, column format
        target = np.array([target[1], target[0], target[3], target[2]])
        # Resize target to range 0-1
        target = np.array([
            target[0] / h, target[1] / w,
            target[2] / h, target[3] / w
        ])

        if self.cfg['use_att_loss']:
            rstarget=target*self.cfg.resize_img[0]
            iou_annot_stage_0=generate_iou_groundtruth([self.cfg.resize_img[0]//8,self.cfg.resize_img[0]//8],
                                                       [(rstarget[0]+rstarget[2])/16,(rstarget[1]+rstarget[3])/16],
                                                       [(rstarget[2]-rstarget[0])/16,(rstarget[3]-rstarget[1])/16])
            iou_annot_stage_1=cv2.resize(iou_annot_stage_0,(self.cfg.resize_img[0]//16, self.cfg.resize_img[1]//16))
            iou_annot_stage_2 = cv2.resize(iou_annot_stage_0, (self.cfg.resize_img[0] // 32, self.cfg.resize_img[1] // 32))
            # cv2.imwrite('./samples/' + str(idx) + '_0.jpg', iou_annot_stage_0*255)
            # cv2.imwrite('./samples/' + str(idx) + '_1.jpg', iou_annot_stage_1*255)
            # cv2.imwrite('./samples/' + str(idx) + '_2.jpg', iou_annot_stage_2*255)
            #
            # # cv2.circle(gttttt, tuple(annot[:2]), 2, 255)
            # # cv2.circle(gttttt, tuple(annot[2:]), 2, 255)
            # cv2.rectangle(img_,tuple(annot[:2]), tuple(annot[2:]),255,10)
            # cv2.imwrite('./samples/' + str(idx) + '_gt.jpg', img_)
        else:
            iou_annot_stage_0 = np.zeros([1])
            iou_annot_stage_1 = np.zeros([1])
            iou_annot_stage_2 = np.zeros([1])
        # Target in range -1 to 1
        target = 2 * target - 1

        # img = self.img_transforms(img)
        # img = Image(pil2tensor(img, np.float_).float().div_(255))
        img = pil2tensor(img, np.float_).float().div_(255)
        out = {
            'img': img,
            'idxs': torch.tensor(idx).long(),
            'qvec': torch.from_numpy(q_chosen_emb_vecs),
            'qlens': torch.tensor(qlen),
            'annot': torch.from_numpy(target).float(),
            'orig_annot': torch.tensor(annot).float(),
            'img_size': torch.tensor([h, w]),
            'iou_annot_stage_0':torch.tensor(iou_annot_stage_0).float(),
            'iou_annot_stage_1': torch.tensor(iou_annot_stage_1).float(),
            'iou_annot_stage_2': torch.tensor(iou_annot_stage_2).float()
        }

        return out

    def load_annotations(self, idx):
        annotation_list = self.image_data.iloc[idx]
        img_file, x1, y1, x2, y2, queries = annotation_list
        img_file = self.img_dir / f'{img_file}'
        if isinstance(queries, list):
            query_chosen = np.random.choice(queries)
        else:
            assert isinstance(queries, str)
            query_chosen = queries
        if '_' in query_chosen:
            query_chosen = query_chosen.replace('_', ' ')
        # annotations = np.array([y1, x1, y2, x2])
        annotations = np.array([x1, y1, x2, y2])
        return img_file, annotations, query_chosen

    def _read_annotations(self, trn_file):
        trn_data = pd.read_csv(trn_file)
        trn_data['bbox'] = trn_data.bbox.apply(
            lambda x: ast.literal_eval(x))
        sample = trn_data['query'].iloc[0]
        if sample[0] == '[':
            trn_data['query'] = trn_data['query'].apply(
                lambda x: ast.literal_eval(x))

        trn_data['x1'] = trn_data.bbox.apply(lambda x: x[0])
        trn_data['y1'] = trn_data.bbox.apply(lambda x: x[1])
        trn_data['x2'] = trn_data.bbox.apply(lambda x: x[2])
        trn_data['y2'] = trn_data.bbox.apply(lambda x: x[3])
        if self.ds_name == 'flickr30k':
            trn_data = trn_data.assign(
                image_fpath=trn_data.img_id.apply(lambda x: f'{x}.jpg'))
            trn_df = trn_data[['image_fpath',
                               'x1', 'y1', 'x2', 'y2', 'query']]
        elif self.ds_name == 'refclef':
            trn_df = trn_data[['img_id',
                               'x1', 'y1', 'x2', 'y2', 'query']]
        elif 'flickr30k_c' in self.ds_name:
            trn_data = trn_data.assign(
                image_fpath=trn_data.img_id.apply(lambda x: x))
            trn_df = trn_data[['image_fpath',
                               'x1', 'y1', 'x2', 'y2', 'query']]
        return trn_df


def collater(batch):
    qlens = torch.Tensor([i['qlens'] for i in batch])
    max_qlen = int(qlens.max().item())
    # query_vecs = [torch.Tensor(i['query'][:max_qlen]) for i in batch]
    out_dict = {}
    for k in batch[0]:
        out_dict[k] = torch.stack([b[k] for b in batch]).float()
    out_dict['qvec'] = out_dict['qvec'][:, :max_qlen]

    return out_dict


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return NewDistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def get_dataloader(cfg, dataset: Dataset, is_train: bool) -> DataLoader:
    is_distributed = cfg.do_dist
    images_per_gpu = cfg.bs
    if is_distributed:
        # DistributedDataParallel
        batch_size = images_per_gpu
        num_workers = cfg.nw
    else:
        # DataParallel
        batch_size = images_per_gpu * cfg.num_gpus
        num_workers = cfg.nw * cfg.num_gpus
    if is_train:
        shuffle = True
    else:
        shuffle = False if not is_distributed else True
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    return DataLoader(dataset, batch_size=batch_size,
                      sampler=sampler, drop_last=is_train,
                      num_workers=num_workers, collate_fn=collater)


def get_data(cfg):
    # Get which dataset to use
    ds_name = cfg.ds_to_use

    # Training file
    trn_csv_file = cfg.ds_info[ds_name]['trn_csv_file']
    trn_ds = ImgQuDataset(cfg=cfg, csv_file=trn_csv_file,
                          ds_name=ds_name, split_type='train')
    trn_dl = get_dataloader(cfg, trn_ds, is_train=True)

    # Validation file
    val_csv_file = cfg.ds_info[ds_name]['val_csv_file']
    val_ds = ImgQuDataset(cfg=cfg, csv_file=val_csv_file,
                          ds_name=ds_name, split_type='valid')
    val_dl = get_dataloader(cfg, val_ds, is_train=False)

    test_csv_file = cfg.ds_info[ds_name]['test_csv_file']
    test_ds = ImgQuDataset(cfg=cfg, csv_file=test_csv_file,
                           ds_name=ds_name, split_type='valid')
    test_dl = get_dataloader(cfg, test_ds, is_train=False)

    data = DataWrap(path=cfg.tmp_path, train_dl=trn_dl, valid_dl=val_dl,
                    test_dl={'test0': test_dl})
    return data


if __name__ == '__main__':
    cfg = conf
    data = get_data(cfg, ds_name='refclef')
