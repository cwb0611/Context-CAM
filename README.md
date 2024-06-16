# Context-CAM: Using multi-scale context to eliminate the background noise regions in the class activation map

In this work, we propose a new method called Context-CAM, which eliminates background noise by using context information from  class activation maps at different scales. 

<img src="pics/pipeline.jpg" width="1200px" height="450px" title="FG-CAM pipeline" alt="FG-CAM pipeline"></img><br/>

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

