# ObjectDetection_KerasRetinanet

This repository contain files for setting up keras-retinanet and codes for training and testing models.
More infomation can be found at the follow repository: https://github.com/fizyr/keras-retinanet.git

# Steps for keras-retinanet single object detection model training and testing

    1. Data pre-processing: maually label images in MATLAB or other tools(so far two ways, one generates B&W images the other generates.xml file); image cropping, if needed(keras-retinanet defaultly resize input image to 800*800 pixels, max 800*1333 pixels); create csv file for training, no need for testing) from the file generated
    2. Check anchors(optional): if target objects are of different sizes and shapes, may check if the current anchors' ratios fits the objects in training set images or not
    3. Model training: customize training parameters, modify input csv files and label name(s), change model name for saving proper, start training
    4. Model testing: modify input folder name and output folder name, customize detecting threshold and etc., model testing
   

Things for noticing during customization
1. When customize anchor box aspecrt ratios, also modify the following file: keras_retinanet/utils/anchors.py




