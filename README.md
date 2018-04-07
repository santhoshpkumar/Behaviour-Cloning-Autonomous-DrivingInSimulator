# Behaviour Cloning - Autonomous Driving in Simulator
## Behavioural Cloning Project


[image1]: ./examples/model_drive.png "Model drive"

This project will build a model to drive a car in a simulated circuit. The training data is obtained by driving the car manually in the simulator and collecting the steering angle and measurementa along with the images (center, left and right). We will then build models to output the mesaruement based on the location captured when driving autonomously. In autonomous mode the images are fed to the model and the correspnding measurements obtained as output will be applied to drive the car.

At first I collected the data by driving the car on the center of the road without much mistakes. This amounted to 900 mb of data. I drove couple of laps and got bored and drove the last few laps in the opposite direction of the track.

Here is the video obtained by stiching all the images in the training set obtained.

Track1:

[![TRACK 1](https://img.youtube.com/vi/bIOxROYhmh0/1.jpg)](https://www.youtube.com/watch?v=bIOxROYhmh0)

Track2:

[![TRACK 2](https://img.youtube.com/vi/LRuZ5QlC0tk/1.jpg)](https://www.youtube.com/watch?v=LRuZ5QlC0tk)

At first will build a generic model where the output is the steering angle based on the image. I will not be using AWS instance to build and train the model but instead will train it on my GTX 1080 Ti. Will record the amount of time taken to build and train the model.

## MODEL 1

This is a simple linear regression model, which outputs the expected measurement for the given image or location on the map.

[![IMAGE_VIDEO](https://img.youtube.com/vi/E5XF0RpSkrI/1.jpg)](https://www.youtube.com/watch?v=E5XF0RpSkrI)

It performs very badly as seen in the video. The car veers everywhere

## MODEL 2

In this we are going to normalize the input image. In this model, a lambda layer is a convenient way to parallelize image normalization. The lambda layer will also ensure that the model will normalize input images when making predictions in drive.py.

That lambda layer could take each pixel in an image and run it through the formulas:
```
pixel_normalized = pixel / 255

pixel_mean_centered = pixel_normalized - 0.5
```
A lambda layer will look something like:
```
Lambda(lambda x: (x / 255.0) - 0.5)
```

[![IMAGE_VIDEO](https://img.youtube.com/vi/PHPITs18BjM/1.jpg)](https://www.youtube.com/watch?v=PHPITs18BjM)

Well it is still no where close to autonomous driving. Time to try the famous LeNet and see how it performs.

## MODEL 3

In this model we will build a CNN for the images we got. This model follows LeNet architecture.

[![IMAGE_VIDEO](https://img.youtube.com/vi/Zi3V4CIJxEo/1.jpg)](https://www.youtube.com/watch?v=Zi3V4CIJxEo)

Seems better but again it still fails to stay on the road, long way to go to get it perfected.

## MODEL 4

We going to augment our images by flipping images and steering measurements. A effective technique involves flipping images and taking the opposite sign of the steering measurement. 

Flipped Images

[![IMAGE_VIDEO](https://img.youtube.com/vi/cW4aDtf9W-g/1.jpg)](https://www.youtube.com/watch?v=cW4aDtf9W-g)

Seems to drive, but get stuck by the time it reaches the bridge.

## MODEL 5

We will now try and utilize the left and right view image along with the center image we have been using so far. It is  possible to use all three camera images to train the model. When recording, the simulator will simultaneously save an image for the left, center and right cameras. Each row of the csv log file, driving_log.csv, contains the file path for each camera as well as information about the steering measurement, throttle, brake and speed of the vehicle. We will now augment our trainign data to contiant these images.

We will feed the left and right camera images to your model as if they were coming from the center camera. This way, we can teach your model how to steer if the car drifts off to the left or the right. Figuring out how much to add or subtract from the center angle will involve some experimentation.

During prediction (i.e. "autonomous mode"), we only predict with the center camera image.

[![IMAGE_VIDEO](https://img.youtube.com/vi/YuEtP6lbjP0/1.jpg)](https://www.youtube.com/watch?v=YuEtP6lbjP0)

## MODEL 6

We will now crop the image and build a area of interest. The modle reads the noise in the image affectign the final accurancy for pour autonomous driving.

On the first trained model with 5 epochs the car was able to cross over the bridge, but failed to contineu forward and went into the lake. The model needs some more tweeking.

On retraining with a better data set, that included lot of images where the steer angle would bring it back to center of the road. With this change was able to drive around the circuit without going off the road.

[![IMAGE_VIDEO](https://img.youtube.com/vi/3gFvRcn0bko/1.jpg)](https://www.youtube.com/watch?v=3gFvRcn0bko)

## MODEL 7

With the armed insight that a better data with equal negative and positive will help produce a better model, I first focused on getting a new data set with images from the second track. 
Also went ahead and used the nvida model and some slight modification of adding a dropoff resulted in a model which now can steer the car even when it veers off road. I am quite satisfied with the modle and a great learning on the effect of the right quality of data to build a model.

[![IMAGE_VIDEO](https://img.youtube.com/vi/_tywEv0Vhno/1.jpg)](https://www.youtube.com/watch?v=_tywEv0Vhno)

All intermediate models fom 1 to 6 can be downloaded [here](https://drive.google.com/open?id=1idwEgE6L2gjdWh97g0cknD7WOlbrhOHS)

## MODEL 8

Post review with subject matter experts it was suggested to add a activation function to each of the fully connected layers. I experiments with different activation functions as to sigmoid, tanh, relu, but what gave the most smooth drive on autonomous mode was the elu (exponential linear unit). Model 7 did lot of sharp turns, but with these activation added I could see smoothing of the turns and car sticking to a virtual lane. While I stuck to my initial 10 epoch, I saw that the models post 4 epoch had lower training error but the validation error increased, indicating overfitting. I saved the model at each epoch and have used the 4 epoch saved parameters as my final model.

Autonomous drive around the track with constant speed of 9 MPH

[![IMAGE_VIDEO](https://img.youtube.com/vi/O6aZjh_V-II/1.jpg)](https://www.youtube.com/watch?v=O6aZjh_V-II)

Autonomous drive around the track with constant speed of 15 MPH 


[![IMAGE_VIDEO](https://img.youtube.com/vi/Uy9pEDzrLtU/1.jpg)](https://www.youtube.com/watch?v=Uy9pEDzrLtU)


[![IMAGE_VIDEO](https://img.youtube.com/vi/hnyZgaBFwic/1.jpg)](https://www.youtube.com/watch?v=hnyZgaBFwic)
