# Classic-CNN-
This is a description of classic convolutional neural network and its implementation in Pytorch <br>
We briefly introduce the structure and #params and FLOPs of these network

## 1. LeNet
LeNet, proposed in 1998 by LeCun, contains 7 layers including 2 conv, 2 pooling and 3 fc layers
![image](https://github.com/MonkeyKing-KK/Classic-CNN-/blob/master/pictures/LeNet.png)
![image](https://github.com/MonkeyKing-KK/Classic-CNN-/blob/master/pictures/LeNet_data.png)

## 2. AlexNet
AlexNet, proposed in 2012, contains 8 layers including 5 conv and 3 fc layers. <br>
### tricks
ReLU <br>
Dropout <br>
Local Response Normalization <br>
Overlapping Pooling

![image](https://github.com/MonkeyKing-KK/Classic-CNN-/blob/master/pictures/Alexnet.png)
![image](https://github.com/MonkeyKing-KK/Classic-CNN-/blob/master/pictures/AlexNet_data.png)

## 3. VGG16
VGG, proposed in 2014, is a series of neural network and contains 16-19 layers. <br>
We only implement VGG16 which consists of 13 conv, 5 pooling and 3 fc layers
### tricks
Deeper <br>
smaller kernel_size
![image](https://github.com/MonkeyKing-KK/Classic-CNN-/blob/master/pictures/VGG16.png)
![image](https://github.com/MonkeyKing-KK/Classic-CNN-/blob/master/pictures/VGG16_data.png)

## 4. SqueezeNet
SqueezeNet, proposed in 2016, is a lightweight neural network. <br>
(reimplementation)
Total param:1,244,448 <br>
MAdd: 1.67G <br>
Flops: 838.94M <br>
### tricks
1. Replace 3x3 filters with 1x1 filters <br>
2. Decrease the number of input channels to 3x3 filters <br>
3. Downsample late in the network so that convolution layers have large activation maps
![image](https://github.com/MonkeyKing-KK/Classic-CNN-/blob/master/pictures/firemodule.png)
