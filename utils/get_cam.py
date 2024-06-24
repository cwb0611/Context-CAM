import cv2
import torch 
from torch import nn
import torchvision.models as models
import argparse

from Model.Context_CAM import *



def get_cam_method(args,net,data_dic):
	if args.test_method in ["Context_CAM","Context_CAM_D"]:
		layer_list = ["features_0","features_2","features_4",
						  "features_5","features_7","features_9",
						  "features_10","features_12","features_14","features_16",
						  "features_17","features_19","features_21","features_23",
						  "features_24","features_26","features_28","features_30"]
		model_dict = dict(type='vgg', arch=net, layer_name=layer_list)
		vis_model = Context_CAM(model_dict)
		return vis_model



def get_cam(args,image,vis_model,class_index,net):
	if args.test_method in ["Context_CAM"]:
		context_cam, logic = vis_model(image,class_idx=class_index,svd_denoise=False)
		context_cam = (context_cam-context_cam.min())/(context_cam.max()-context_cam.min()+1e-8)
		return context_cam

	if args.test_method in ["Context_CAM_D"]:
		context_cam, logic = vis_model(image,class_idx=class_index,svd_denoise=True)
		context_cam = (context_cam-context_cam.min())/(context_cam.max()-context_cam.min()+1e-8)
		return context_cam
