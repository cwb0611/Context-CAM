

import os, sys, time, random                            
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch 

from tqdm import tqdm
import argparse

from utils.get_data import *
from utils.get_cam import *
from utils.save_cam import *
from utils.get_net import *

import time



def create_path(path):
    if not(os.path.exists(path)):
        os.mkdir(path)
    return

def get_args():
    parser = argparse.ArgumentParser(description='The Pytorch code of Context-CAM')
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--dataset_name", type=str, default="ISLVRC2012")
    parser.add_argument("--test_net", type=str, default="vgg16")
    parser.add_argument("--test_method", type=str, default="Context_CAM")
    args = parser.parse_args()

    create_path("./cam_output/"+str(args.test_method)+"/")

    return args

def main():
    args = get_args()
    print(args)
    
    data_loader, data_dic = get_data(args)
    net = get_net_for_cam(args,data_dic)
    vis_model = get_cam_method(args,net,data_dic)

    loop = tqdm(enumerate(data_loader), total =len(data_loader))
    for j, batch in loop:
        image = batch[0].cuda()
        labels = batch[1].cuda()
        box = batch[2].numpy().tolist()
        name =  batch[3][0]
        image_size = batch[4]
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        denormalized = image.clone()
        for channel, mean, std in zip(denormalized[0], means, stds):
            channel.mul_(std).add_(mean)
        ori_image = denormalized.squeeze(0).permute(1,2,0)

        for class_index in labels:
            context_cam = get_cam(args,image,vis_model,class_index,net)
            print(context_cam.sum())
            vis_module(ori_image, context_cam[0].cpu().detach().numpy(),name+"_"+str(class_index.item()), "./cam_output/"+str(args.test_method)+"/")
    print(args.dataset_name)
    print(args.test_method)


if __name__ == "__main__":
    main()
