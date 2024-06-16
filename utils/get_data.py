import torch 
from torch.utils.data import DataLoader
import torchvision.transforms as T
import argparse
import utils.dataset

from torchvision.transforms import InterpolationMode

ISLVRC2012_dic ={"root":'./data/ISLVRC2012/',
			     "image_size":int(256),
			     "crop_size":int(224),
			     "class_num":int(1000)
			      }


def get_data(args):
	if args.dataset_name == "ISLVRC2012" :
		data_dic = ISLVRC2012_dic
		imagenet_val = utils.dataset.ISLVRC2012_Dataset(data_dic["root"], 
                              'cls/val_list1.txt',
                              'box/val_box.mat',image_transform=utils.dataset.ImageClassification(crop_size=data_dic["crop_size"]))

		data_loader = DataLoader(imagenet_val, batch_size = args.batch_size)
		return data_loader,data_dic