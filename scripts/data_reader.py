#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


get_ipython().run_line_magic('matplotlib', 'notebook')
cap = cv2.VideoCapture('./phase2/rgb_videos/DJI_0001_2.mov')
index = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
#     frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    if index%150==0:
        cv2.imwrite('./phase2/rgb_frames/DJI_0001_2/DJI_0001_2_' + str(int(index/150)) + '.jpg', frame)
#     plt.imshow(frame)
    index += 1
cap.release()


# In[2]:


import time
t1 = time.time()
print(t1)
cap = cv2.VideoCapture('./phase2/rgb_videos/DJI_0001_2.mov')
index = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    cv2.imwrite('./temp/' + str(index) + '.jpg', frame)
    index += 1
cap.release()
t2 = time.time()
print(t2)
print(t2 - t1)


# image size: 1920 X 1080
# Number of Frames: 29.72*(16*60 + 37) = 29, 630 frames
# Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz,  ran on one core, under 1 GB ram usage
# Total time: 1437.3455204963684
#     

# In[ ]:




