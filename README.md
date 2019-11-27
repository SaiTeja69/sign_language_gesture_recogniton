# sign_language_gesture_recogniton
to recognize gestures made by hand , images classified using squeezenet

Using Canny Edge detection to clear out the noise from the images .
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg',0)
edges = cv2.Canny(img,100,200)

To train your own nn 
Go to res-> image_collector.py 
Change the class map in train.py and test.py suitably 
then run train.py.

To test your model , run test.py
