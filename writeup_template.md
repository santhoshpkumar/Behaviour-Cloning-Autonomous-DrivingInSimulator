# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/architecture.png "Nvidia Architecture"
[image2]: ./examples/placeholder.png "Grayscaling"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on Nvidia Network architecure used in their self driving car. I have added a additional dorpout layer between the fully connected layer.

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting  

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and data from second track to generalize. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try the approach presented in the class exercises. Start with basic logistic model, progress to LeNet and finally Nvidia Architecture.

Details of the model can be accessed here: https://github.com/santhoshpkumar/BehaviourCloningAutonomousDrivingInSimulator/blob/master/README.md

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle veered towards the edge of the track and corrected back to center.

Model 6 with LeNet worked pretty well without issues around the track and Model 7 was based on the Nvidai architecuture which had same resutls as mode 6, but showed more responsive behavior at increased speed (tweeked the drive.py and set the default to higher value than 9)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Data set: https://drive.google.com/open?id=1MlVImnsCtx-FI82rfXBwv6mLrP7Tdodg

One great learning from this project was that, quality of the data is utmost important as compared to that of model importance. I spend time tweeking and experimenting multiple model with the default provided dataset, I even created 10 laps of properly driven behavior images. What finally hit me was tht the model will never perform well if it does nto have the data set of bad behvior, that is the most critical to get the car back on the road. Track 2 helps to generalize the data but the images still does not provide enough training for the model to drive with in the track limits.

To capture good driving behavior, I recorded one laps on track one using center lane driving. The video link has 3 laps. The last lap was done driving in the center of the track

I have recorded the the first two laps where in the vehichle is recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... The easiest way to generate such data was to drive in a zig-zag pattern both clockwise and anticlockwise.


https://youtu.be/bIOxROYhmh0

Then I added more recording of driving on one side of the track two in order to get more data points.

https://youtu.be/LRuZ5QlC0tk

To augment the data sat, I also flipped each of the images and angles thinking that this would add more training images. I have set the measurment correction to +/- 0.30 so that it manevour off to the center of the road if the image matches that of the left or the right image.

I finally randomly shuffled the data set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.  The ideal number of epochs was 10. However I noticed that the model at the final epoc has much higher validatoin loss as compared to the model before it, so added steps to save the model at each epoch and pick the one that had lowest validation loss in last coupel of epochs. 

### Final Video Link: https://youtu.be/_tywEv0Vhno
