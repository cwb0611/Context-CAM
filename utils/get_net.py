
import cv2
import torch 
from torch import nn
import torchvision.models as models
import argparse


def get_net_for_cam(args,data_dic):
	if args.test_net == "vgg16":
		if args.dataset_name in ["ISLVRC2012"]:
			net =  models.vgg16(pretrained=True)
			net = net.cuda().eval()
			return net
