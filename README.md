# Context-CAM: Context-Level Weight-Based CAM with Sequential Denoising to Generate High-Quality Class Activation Maps

In this work, an innovative Context-level weights-based CAM (Context-CAM) method is proposed, which guarantees: 1) the non-discriminative regions that have similar appearances with and are located close to the discriminative regions can be highlighted as well by the newly designed Region-Enhanced Mapping (REM) module using context-level weights; and 2) the background noises are gradually eliminated via a newly proposed Semantic-Guided Reverse Sequence Fusion (SRSF) strategy that can sequentially denoise and fuse the region-enhanced maps from the last layer to the first layer.

This github shows the visualization of Context-CAM and the recent [FG-CAM](https://github.com/dongmo-qcq/FG-CAM) method (AAAI 2024), as shown below:

<div align=center>
<img src="paper_image/fig8.jpg" width="80%" height="80%" title=" Visualization results of FG-Grad-CAM, FG-Grad-CAMsup>D</sup>, FG-Score-CAM, FG-Score-CAMsup>D</sup>, Context-CAM, and Context-CAMsup>D</sup>"></img><br/>
</div>

## Some issues

1.In this github, we only release the code for ISLVRC2012 data, and the codes for other datasets are being organized.

2.We provide some images for testing. If you need to test more images, you need to go to [ISLVRC2012](https://image-net.org/index.php) to download more images, and put image into path "./data/ISLVRC2012/image/".

3.The output CAM visualization will be saved in the folder "cam_output"

### Example commands
```
python3 -W ignore main.py --test_method "Context_CAM"     for testing the Context-CAM method
python3 -W ignore main.py --test_method "Context_CAM_D"   for testing the Context-CAM method with SVD denoising
```

## Note
Thanks to [Haofan Wang](https://github.com/haofanwang/Score-CAM) and [mhyatt00](https://github.com/mhyatt000/layerCAM). 
The format of this code is borrowed from [Score-CAM](https://github.com/haofanwang/Score-CAM) and  [LayerCAM](https://github.com/mhyatt000/layerCAM).

