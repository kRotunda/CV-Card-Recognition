Card Recognition Program
This program uses the computer camera to search for a playing card based on edge detection.
When it detects a card, the program runs the frame of the video through two convolutional neural network (CNN)
models to make a prediction on the card's suit and value, which are then displayed on the screen.

How it Works
The program uses OpenCV to capture frames from the computer camera and perform edge detection on the frames.
It then uses contour detection to find rectangular shapes in the image, which are assumed to be playing cards.
For each detected card, the program extracts the card image from the frame and resizes it to 200 x 200 pixels.
Next, the program runs the card image through two CNN models to predict its suit and value.
The CNN models were trained on a dataset of playing card images that were created using another program that
also used edge detection to find rectangular shapes in the image.
Finally, the program displays the predicted suit and value on the screen.

Dependencies
The program requires the following dependencies to be installed:
Python 3.7 or later
OpenCV
TensorFlow
Keras

How to Use
Clone the repository to your local machine.
Open a terminal and navigate to the project directory.
Run the card_recognition.py script using Python:
Hold a playing card up to the computer camera and wait for the program to detect it.
The program will display the predicted suit and value on the screen.