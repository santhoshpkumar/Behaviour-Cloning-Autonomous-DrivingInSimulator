# Behaviour Cloning - Autonomous Driving in Simulator
## Behavioural Cloning Project


[image1]: ./examples/model_drive.png "Model drive"

This project will build a model to drive a car in a simulated circuit. The training data is obtained by driving the car manually in the simulator and collecting the steering angle and measurementa along with the images (center, left and right). We will then build models to output the mesaruement based on the location captured when driving autonomously. In autonomous mode the images are fed to the model and the correspnding measurements obtained as output will be applied to drive the car.

At first I collected the data by driving the car on the center of the road without much mistakes. This amounted to 900 mb of data. I drove couple of laps and got bored and drove the last few laps in the opposite direction of the track.

Here is the video obtained by stiching all the images in the training set obtained.

Track1:
https://youtu.be/bIOxROYhmh0

Track2:
https://youtu.be/LRuZ5QlC0tk

At first will build a generic model where the output is the steering angle based on the image. I will not be using AWS instance to build and train the model but instead will train it on my GTX 1080 Ti. Will record the amount of time taken to build and train the model.

## MODEL 1

This is a simple linear regression model, which outputs the expected measurement for the given image or location on the map.

![alt text][image1]

Click Here for video -- >(https://youtu.be/E5XF0RpSkrI)

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

![alt text][image1]

Click Here for video -- >(https://youtu.be/PHPITs18BjM)

Well it is still no where close to autonomous driving. Time to try the famous LeNet and see how it performs.

## MODEL 3

In this model we will build a CNN for the images we got. This model follows LeNet architecture.

![alt text][image1]

Click Here for video -- >(https://youtu.be/Zi3V4CIJxEo)

Seems better but again it still fails to stay on the road, long way to go to get it perfected.

## MODEL 4

We going to augment our images by flipping images and steering measurements. A effective technique involves flipping images and taking the opposite sign of the steering measurement. 

Flipped Images

![alt text][image1]

Click Here for video -- >(https://youtu.be/cW4aDtf9W-g)

Seems to drive, but get stuck by the time it reaches the bridge.

## MODEL 5

We will now try and utilize the left and right view image along with the center image we have been using so far. It is  possible to use all three camera images to train the model. When recording, the simulator will simultaneously save an image for the left, center and right cameras. Each row of the csv log file, driving_log.csv, contains the file path for each camera as well as information about the steering measurement, throttle, brake and speed of the vehicle. We will now augment our trainign data to contiant these images.

We will feed the left and right camera images to your model as if they were coming from the center camera. This way, we can teach your model how to steer if the car drifts off to the left or the right. Figuring out how much to add or subtract from the center angle will involve some experimentation.

During prediction (i.e. "autonomous mode"), we only predict with the center camera image.

![alt text][image1]

Click Here for video -- >(https://youtu.be/YuEtP6lbjP0)

## MODEL 6

We will now crop the image and build a area of interest. The modle reads the noise in the image affectign the final accurancy for pour autonomous driving.

On the first trained model with 5 epochs the car was able to cross over the bridge, but failed to contineu forward and went into the lake. The model needs some more tweeking.

On retraining with a better data set, that included lot of images where the steer angle would bring it back to center of the road. With this change was able to drive around the circuit without going off the road.

![alt text][image1]

Click Here for video -- >(https://youtu.be/3gFvRcn0bko)

## MODEL 7

With the armed insight that a better data with equal negative and positive will help produce a better model, I first focused on getting a new data set with images from the second track. 
Also went ahead and used the nvida model and some slight modification of adding a dropoff resulted in a model which now can steer the car even when it veers off road. I am quite satisfied with the modle and a great learning on the effect of the right quality of data to build a model.

![alt text][image1]

Click Here for run1 -- >(https://youtu.be/alTj2xaDM7Y)

Click Here for run2 -- >(https://youtu.be/KJWZdH0xHjs)


All intermediate models fom 1 to 6 can be downloaded here -->> https://drive.google.com/open?id=1idwEgE6L2gjdWh97g0cknD7WOlbrhOHS
