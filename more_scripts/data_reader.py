#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# ### extract frames at particular frequency

# In[6]:


get_ipython().run_line_magic('matplotlib', 'notebook')
cap = cv2.VideoCapture('./phase3/CV4_OpenPit/DJI_0011.mov')
index = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
#     frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    if index%90==0:
        cv2.imwrite('./phase3/rgb_frames/DJI_3011_0/DJI_3011_0_' + format(index//90, '05d') + '.jpg', frame)
#     plt.imshow(frame)
    index += 1
cap.release()


# ### calculate frame rate ratio for IR and RGB video

# In[10]:


cap = cv2.VideoCapture('./phase3/CV3_OpenPit/DJI_0005.mov')
index = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    index += 1
cap.release()
print(index)


# In[8]:


DJI_3002_0_IR_frame_count = 3165
DJI_3002_0_RGB_frame_count = 3136
RGB_to_IR_ratio = DJI_3002_0_RGB_frame_count / DJI_3002_0_IR_frame_count
print(RGB_to_IR_ratio)


# In[11]:


DJI_3005_0_IR_frame_count = 9471
DJI_3005_0_RGB_frame_count = 9386
RGB_to_IR_ratio = DJI_3005_0_RGB_frame_count / DJI_3005_0_IR_frame_count
print(RGB_to_IR_ratio)


# ### calculate time to write the frame to harddisk

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
