import torch
from torch.utils.data import Dataset
import PIL.Image
import scipy.io as scio 
import cv2
import os
import numpy as np
from typing import Optional, Tuple
from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode




def load_img_name_labels_list(cls_path):
    img_gt_name_list = open(cls_path).read().splitlines()
    images = []
    labels = []
    for img_gt_name in img_gt_name_list:
        image, label = img_gt_name.strip().split()
        images.append(image)
        labels.append(int(label))
    return images, labels



def load_box_list(box_path):
    annotations = scio.loadmat(box_path)['rec'][0]
    bboxes = []
    for image_index in range(len(annotations)):
        bbox = []
        for box_index in range(len(annotations[image_index][0][0])):
            xyxy = annotations[image_index][0][0][box_index][1][0]
            bbox.append(xyxy.tolist())
        bboxes.append(bbox)
    return bboxes

def find_test_data(data_list, img_name_list, test_img_name_list):
    test_data_list = []
    for test_img_name in test_img_name_list:
        for i in range(len(img_name_list)):
            if img_name_list[i] == test_img_name:
                test_data_list.append(data_list[i])
    return test_data_list





class ImageClassification(nn.Module):
    def __init__(self, *, crop_size: int, resize_size: int = 256, mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225), interpolation: InterpolationMode = InterpolationMode.BILINEAR,) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, img: Tensor) -> Tensor:
        img = F.resize(img, self.resize_size, interpolation=self.interpolation)
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )


######**************************ISLVRC2012****************************

class ISLVRC2012_Dataset(Dataset):

    def __init__(self, root, img_name_label_list_path, box_path, image_transform=None):
        self.test_img_name_list = os.listdir(root+ "image/")

        img_name_list, label_list = load_img_name_labels_list(root + img_name_label_list_path)
        box_list = load_box_list(root + box_path)

        self.test_label_list = find_test_data(label_list, img_name_list, self.test_img_name_list)
        self.test_box_list = find_test_data(box_list, img_name_list, self.test_img_name_list)
        self.root = root
        self.image_transform = image_transform
        if len(self.test_img_name_list) != len(self.test_box_list):
            print("dataset is error")

    def __len__(self):
        return len(self.test_img_name_list)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.root + "image/" + self.test_img_name_list[idx]).convert("RGB")
        ori_img = np.array(img)
        w, h = img.size[:2]
        image_size = [h, w]
        if self.image_transform:
            img = self.image_transform(img)
        label = torch.tensor(self.test_label_list[idx])
        box = torch.tensor(self.test_box_list[idx])
        return img, label, box, self.test_img_name_list[idx][:-5], image_size, ori_img

