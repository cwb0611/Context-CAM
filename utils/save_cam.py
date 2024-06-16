

import numpy as np
import cv2




def vis_module(ORI_IMAGE, CAM, NAME, SAVE_ROOT, ALPHA=0.6):
    #CAM          N*N*1    0-1
    #ORI_IMAHE    N*N*3    0-1
    #NAME         STR
    #SAVE_ROOT    STR

    heatmap = np.uint8(255 * CAM)
    ORI_IMAGE = cv2.cvtColor((255 * ORI_IMAGE.cpu().numpy()).astype(np.uint8), cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap*ALPHA + ORI_IMAGE*(1-ALPHA)    
    cv2.imwrite(SAVE_ROOT+NAME+".png", superimposed_img)
    return