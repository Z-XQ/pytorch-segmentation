from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ignore_label = 255
# ID_TO_TRAINID = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
#                  3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
#                  7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
#                  14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
#                  18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
#                  28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
from utils.load_json_label import json_to_cls_mask


class DefectDataset(BaseDataSet):
    def __init__(self, mode='fine', label_id_dict={}, **kwargs):
        self.num_classes = len(label_id_dict)  # 背景的id默认是0，所以字典内没有写，总的分类数需要加上背景。
        # self.mode = mode
        # self.palette = palette.CityScpates_palette
        self.label_id_dict = label_id_dict
        super(DefectDataset, self).__init__(**kwargs)

    def _set_files(self):
        image_paths, label_paths = [], []
        for root_, dirs_, files in os.walk(self.root):
            for file_name in files:
                file_ex = file_name.split(".")[-1]
                if file_ex in ["png", "jpg", "jpeg", "bmp", "tif"]:
                    image_path = os.path.join(root_, file_name)
                    json_path = image_path.replace(file_ex, "json")
                    image_paths.append(image_path)
                    label_paths.append(json_path)
        # image_path = os.path.join(self.root, img_dir_name, 'leftImg8bit', self.split)
        assert len(image_paths) == len(label_paths)
        self.files = list(zip(image_paths, label_paths))

    def _load_data(self, index):
        image_path, label_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        # label = np.asarray(Image.open(label_path), dtype=np.int32)  # 载入原始标注的label图像(h,w)
        # for k, v in self.id_to_trainId.items():  # 将label图像中类别，统一映射到其他类别。
        #     label[label == k] = v
        # 读取json文件，不同的类别使用不同的class_id
        label = json_to_cls_mask(label_path, self.label_id_dict)
        return image, label, image_id  # (h,w,3), (h,w), image file name


class Defect(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 mode='fine', val=False,
                 shuffle=False, flip=False, rotate=False, blur=False,
                 augment=False, val_split=None, return_id=False,
                 label_id_dict=None):
        self.MEAN = [0., 0., 0.]
        self.STD = [1., 1., 1.]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }
        # dataset
        self.dataset = DefectDataset(label_id_dict=label_id_dict, **kwargs)
        # dataloader
        super(Defect, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
