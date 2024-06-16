import torch
import torchvision
import torch.nn.functional as F
from Model.grad_base_cam import *
import numpy as np


class Context_CAM(Base_Mult_layer_CAM):                     

    def __init__(self, model_dict):                                      
        super().__init__(model_dict)                     
        self.layer_num = len(model_dict['layer_name'])
        self.region_enhanced_maps = None
        self.denoised_maps = None
        self.binary_maps = None
        self.attention_maps = None

    def forward(self, input, class_idx=None, svd_denoise = False, retain_graph=False):         
        b, c, h, w = input.size()

        self.region_enhanced_maps = torch.zeros((b,self.layer_num,h,w)).cuda()
        self.denoised_maps = torch.zeros((b,self.layer_num,h,w)).cuda()
        self.binary_maps = torch.ones((b,self.layer_num+1,h,w)).cuda()
        self.attention_maps = torch.ones((b,self.layer_num+1,h,w)).cuda()

        logit = self.model_arch(input).cuda()                             

        if class_idx is None:                                            
            predicted_class = logit.max(1)[-1]                            
        else:
            predicted_class = torch.LongTensor([class_idx])               
                                        
        if torch.cuda.is_available():                                               
          predicted_class = predicted_class.cuda()
          logit = logit.cuda()
        
        one_hot_output = torch.FloatTensor(1, logit.size()[-1]).zero_()            
        one_hot_output[0][predicted_class] = 1                               
        one_hot_output = one_hot_output.cuda(non_blocking=True)               
        # Zero grads
        self.model_arch.zero_grad()                                            
        # Backward pass with specified target
        logit.backward(gradient=one_hot_output)                                
        self.activations.reverse()                     # Feature maps are arranged from deep to shallow

        with torch.no_grad():
            for i in range(self.layer_num):
                horizontal_weights = self.gradients[i].max(dim = 2, keepdim = True)[0]
                vertical_weights = self.gradients[i].max(dim = 3, keepdim = True)[0]
                rem = self.activations[i] * (vertical_weights + horizontal_weights)
                if svd_denoise == True :
                    rem = self.svd(rem)
                rem = torch.sum(rem, dim=1).unsqueeze(0) 
                rem = F.interpolate(rem, size=(h, w), mode='bilinear', align_corners=False)
                rem = self.normalized(rem)                  
                self.region_enhanced_maps[0][i] = rem
                self.attention_maps[0][i] = self.binary_maps[:,:i+1].sum(1)
                self.denoised_maps[0][i] = self.region_enhanced_maps[0][i]*self.attention_maps[0][i]
                self.denoised_maps[0][i] = self.normalized(self.denoised_maps[0][i])
                self.binary_maps[0][i] = self.Adaptive_Binarization(self.denoised_maps[0][i])
            ba_cam = self.denoised_maps.sum(1)/self.layer_num
        self.remove_feature_and_gradients()
        return ba_cam, logit


    def __call__(self, input, class_idx=None, svd_denoise = False, retain_graph=False):
        return self.forward(input, class_idx, svd_denoise, retain_graph)

    def Adaptive_Binarization(self, CAM, threshold_num = 50):
        gray_mean_0 = torch.zeros(threshold_num, dtype=torch.float)
        gray_mean_1 = torch.zeros(threshold_num, dtype=torch.float)
        w0 = torch.zeros(threshold_num, dtype=torch.float)
        w1 = torch.zeros(threshold_num, dtype=torch.float)
        threshold = torch.linspace(CAM[CAM!=CAM.min()].min(),
                                   CAM[CAM!=CAM.max()].max(),steps=threshold_num)
        y_size, x_size = CAM.shape[-2:]
        for i in range(int(threshold_num*0.1), int(threshold_num*0.9)):
            w0[i] = (CAM > threshold[i]).sum() / (y_size*x_size)
            gray_mean_0[i] = torch.mean(CAM[CAM > threshold[i]])
            gray_mean_1[i] = torch.mean(CAM[CAM <= threshold[i]])
            w1 = 1 - w0 
        var = w0*w1*(gray_mean_0-gray_mean_1)**2    
        best_threshold = threshold[torch.argmax(var)]
        binary_cam = torch.where(CAM > best_threshold, 
                                 torch.ones_like(CAM),
                                 torch.zeros_like(CAM))
        return binary_cam

    def svd(self,I):
        b, c, w, h = I.shape
        if w * h  >= 224 * 224:
            I = F.interpolate(I, size=(int(w*3/4), int(h*3/4)), mode='bilinear', align_corners=False)
            torch.cuda.empty_cache()
            I = torch.nan_to_num(I[0])
            reshaped_I = (I).reshape(c, -1)
            reshaped_I= reshaped_I - reshaped_I.mean(dim=1)[:,None]
            U, S, VT = torch.linalg.svd(reshaped_I, full_matrices=True)
            d = 2 
            s = torch.diag(S[:d],0)
            new_I = U[:,:d].mm(s).mm(VT[:d,:])
            new_I = new_I.reshape(I.size()).unsqueeze(0)
            I_d = F.interpolate(new_I, size=(int(w*4/3), int(h*4/3)), mode='bilinear', align_corners=False)
        else:
            torch.cuda.empty_cache()
            I = torch.nan_to_num(I[0])
            reshaped_I = (I).reshape(c, -1)
            reshaped_I= reshaped_I - reshaped_I.mean(dim=1)[:,None]
            U, S, VT = torch.linalg.svd(reshaped_I, full_matrices=True)
            d = 2 
            s = torch.diag(S[:d],0)
            new_I = U[:,:d].mm(s).mm(VT[:d,:])
            I_d = new_I.reshape(I.size()).unsqueeze(0)
        return I_d

    def normalized(self,M):
        return (M-M.min())/(M.max()-M.min()+1e-8)              