'''
Part of code borrows from https://github.com/1Konny/gradcam_plus_plus-pytorch
'''

import torch


class Base_Mult_layer_CAM(object):                         
    """ A framework for obtaining features and associated gradients from multiple convolutional layers.

        : Args
            - **model_dict -** : Dict. Has format as dict(type='vgg', arch=torchvision.models.vgg16(pretrained=True),
            layer_name='features',input_size=(224, 224)).

    """
    def __init__(self, model_dict):                   
        model_type = model_dict['type']                          
        layer_name = model_dict['layer_name']     
        
        self.model_arch = model_dict['arch']                   
        self.model_arch.eval()                                   
        
        if torch.cuda.is_available():                           
          self.model_arch.cuda()

        self.gradients = list()                                   
        self.activations = list()                                 

        def backward_hook(module, grad_input, grad_output):                      
            if torch.cuda.is_available():                                 
              self.gradients.append(grad_output[0].cuda())
            else:
              self.gradients.append(grad_output[0])
            return None

        def forward_hook(module, input, output):                
            if torch.cuda.is_available():                                      
              self.activations.append(output.cuda())
            else:
              self.activations.append(output)
            return None





        if 'vgg' in model_type.lower():                                                      
            self.target_layer = self.find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            self.target_layer = self.find_resnet_layer(self.model_arch, layer_name)

        for layer in self.target_layer:
            layer.register_forward_hook(forward_hook)             
            layer.register_backward_hook(backward_hook)          

    def forward(self, input, class_idx=None, retain_graph=False):
        return None

    def __call__(self, input, class_idx=None, retain_graph=False):            
        return self.forward(input, class_idx, retain_graph)                   

    def find_vgg_layer(self, arch, target_layer_name):
        target_layer = []
        for name in target_layer_name:
            hierarchy = name.split('_')
            if len(hierarchy) >= 1:
                tem_target_layer = arch.features
            if len(hierarchy) == 2:
                tem_target_layer = tem_target_layer[int(hierarchy[1])]
            target_layer.append(tem_target_layer)

        return target_layer

    def find_resnet_layer(self, arch, target_layer_name):
        target_layer = []
        for name in target_layer_name:
            if 'layer' in name:   
                hierarchy = name.split('_')
                layer_num = int(hierarchy[1]) 
                if layer_num == 1:                                        #处理层
                    tem_target_layer = arch.layer1
                elif layer_num == 2:
                    tem_target_layer = arch.layer2
                elif layer_num == 3:
                    tem_target_layer = arch.layer3
                elif layer_num == 4:
                    tem_target_layer = arch.layer4
                if len(hierarchy) >= 3:                                                                                                         
                    bottleneck_num = int(hierarchy[2])        
                    tem_target_layer = tem_target_layer[bottleneck_num]                                               

                if len(hierarchy) >= 4:                                                            
                    tem_target_layer = tem_target_layer._modules[hierarchy[3]]

                if len(hierarchy) == 5:
                    tem_target_layer = tem_target_layer._modules[hierarchy[4]]
            else:
                tem_target_layer = arch._modules[name]
            target_layer.append(tem_target_layer)

        return target_layer

    def remove_feature_and_gradients(self):
        self.gradients = list()                                   
        self.activations = list()                                
        return