# Behaviour Cloning - Autonomous Driving in Simulator
## Behavioural Cloning Project

This project will build a model to drive a car in a simulated circuit. The training data is obtained by driving the car manually in the simulator and collecting the steering angle and measurementa along with the images (center, left and right). We will then build models to output the mesaruement based on the location captured when driving autonomously. In autonomous mode the images are fed to the model and the correspnding measurements obtained as output will be applied to drive the car.

At first I collected the data by driving the car on the center of the road without much mistakes. This amounted to 900 mb of data. I drove couple of laps and got bored and drove the last few laps in the opposite direction of the track.

Here is the video obtained by stiching all the images in the training set obtained.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE)

At first will build a generic model where the output is the steering angle based on the image. I will not be using AWS instance to build and train the model but instead will train it on my GTX 1080 Ti. Will record the amount of time taken to build and train the model.

[image] gtx1080Ti

## MODEL 1

This is a simple linear regression model, which outputs the expected measurement for the given image or location on the map.

