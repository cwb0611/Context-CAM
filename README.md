# Context-CAM: Using multi-scale context to eliminate the background noise regions in the class activation map

In this work, an innovative Context-level weights-based CAM (Context-CAM) method is proposed, which guarantees: 1) the non-discriminative regions that have similar appearances with and are located close to the discriminative regions can be highlighted as well by the newly designed Region-enhanced Mapping (REM) module with context-level weights; and 2) the background noises are gradually eliminated via a newly proposed Semantic-Guided Reverse Sequence Fusion (SRSF) strategy that can sequentially denoise and fuse the region-enhanced maps from the last layer to the first layer.

<img src="paper_image/fig8.jpg" width="1037px" height="1217px" title=" Visualization results of FG-Grad-CAM, FG-Grad-CAMsup>D</sup>, FG-Score-CAM, FG-Score-CAMsup>D</sup>, Context-CAM, and Context-CAMsup>D</sup>"></img><br/>

## Some issues

1.In this github, we only release the code about ISLVRC2012 data, and the codes for other datasets are being organized.

2.We provide some images for testing. If you need to test more images, you need to go to [ISLVRC2012](https://image-net.org/index.php) to download more images.

3.The output CAM visualization will be saved in the folder "cam_output"

### Example commands
```
python3 main.py --test_method "Context_CAM"     for testing the Context-CAM method
python3 main.py --test_method "Context_CAM_D"   for testing the Context-CAM method with SVD denoising
```

## Note
Thanks to [Haofan Wang](https://github.com/haofanwang/Score-CAM) and [mhyatt00](https://github.com/mhyatt000/layerCAM). The format of this code is borrowed from [Score-CAM](https://github.com/haofanwang/Score-CAM) and  [mhyatt00](https://github.com/mhyatt000/layerCAM).

