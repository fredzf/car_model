# Grab Computer Vision Challenge - Stanford Cars

This repository is to build a model to classify cars models using stanford-cars dataset.


## Requirements

- `tensorflow`
- `numpy`
- `opencv`
- `keras`
- `pandas`


## Steps

1.  Download and extract image data from [stanford cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html);
2.  Run `data processing.ipynb` to pre process the data;
3.  Run `car_model.ipynb` to train the model;
4.  Run `prediction.ipynb` to predict on the testing data.


## Solutiin and Results

The problem is to classify the car models and make using stanford dataset. Unlike traditional image classification problem which is to categorize different objects, such as cats and dogs, trees and cars; this problem is to classify car model and make within the class. In literature, the problem is called Fine-grained image classification probem. The problem is challenging, because the visual differences between the categories are small and can be easily overwhelmed by those minor difference caused by factors such as pose, viewpoint, or location of the object in the image.

### Pre-processing
The data was cropped with the bounding box to reduce background noise. 80/20 rule was applied for train-test split.

#### 1st Try

The first try is to tranfer learn the pre-trained models. VGG16, ResNet50 were experimented. The model results are:

| Architecture        |     Validation Accuracy     |       Testing Accuracy      |
|---------------------|:---------------------------:|:---------------------------:|
| VGG16               |             81%             |             79%             |
| ResNet50            |            84.5%            |             83%             |

Traditional architectures were able to classify cars; however, is not as significant as expected. Inspired by a paper on [Bilinear CNN Models for Fine-grained Visual Recognition](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf), a modified architecture was designed.

#### 2nd Try

##### BCNN
Bilinear CNN[1] is a simple yet powerful model for fine-grained image classification problem. The image is passed through two CNNs, A and B, and their outputs are multiplied using outer product at each location of the image and pooled to obtain the bilinear vector. This is passed through a classification layer to obtain predictions. 
By bilinear pooling, more feature combinations were created so as to classify minor difference between objects easily.
![BCNN](imgs/BCNN.png)

Instead of passing to 2 networks, the model used for this projetc is only passing to 1 VGG16 network to reduce processing time. The model architecture is as following:
![model architecture](imgs/model.png)

The result shows that, bilinear pooling improved the model results significantly, where the validation accuracy increased to 89.2% and testing accuracy increased to 87.8%.

| Architecture        |     Validation Accuracy     |       Testing Accuracy      |
|---------------------|:---------------------------:|:---------------------------:|
| VGG16-bilinear      |            89.2%            |           87.8%             |


## Further improvements


## Reference

[1] [BCNN_keras](https://github.com/tkhs3/BCNN_keras).
[2] [Bilinear CNN Models for Fine-grained Visual Recognition](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf).
