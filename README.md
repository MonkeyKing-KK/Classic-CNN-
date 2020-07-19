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
VGG is a series of neural network and contains 16-19 layers. <br>
We only implement VGG16 which consists of 13 conv, 5 pooling and 3 fc layers
### tricks
Deeper <br>
smaller kernel_size
![image](https://github.com/MonkeyKing-KK/Classic-CNN-/blob/master/pictures/VGG16.png)
![image](https://github.com/MonkeyKing-KK/Classic-CNN-/blob/master/pictures/VGG16_data.png)
