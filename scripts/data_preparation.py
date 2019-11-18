#!/usr/bin/env python
# coding: utf-8

# ### create the trainval and test list for idler dataset

# In[1]:


import os
from random import shuffle


# In[2]:


list_files = os.listdir('/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/JPEGImages/')
shuffle(list_files)
trainval_list = list_files[:100]
test_list = list_files[100:]
print(len(list_files), len(trainval_list), len(test_list))
# print('trainval_list:', trainval_list)
# print('test list:', test_list)


# In[2]:


trainval_list = os.listdir('/home/ubuntu/gfav/data/phase2/rgb_frames_int/DJI_0002_0/')
# train_list = [element[:-4] for element in train_list_with_suffix]
test_list = os.listdir('/home/ubuntu/gfav/data/phase2/rgb_frames_int/DJI_0001_0/')
# test_list = [element[:-4] for element in test_list_with_suffix]
# print(train_list, test_list)


# In[3]:


with open('/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/ImageSets/Main/trainval.txt', 'w') as f:
    for item in trainval_list:
        try:
            int(item[:-4])
        except:
            continue
        f.write("%s\n" % item[:-4])
with open('/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/ImageSets/Main/test.txt', 'w') as f:
    for item in test_list:
        try:
            int(item[:-4])
        except:
            continue
        f.write("%s\n" % item[:-4])


# ### visualize labels

# In[1]:


import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


# In[2]:


list_files = os.listdir('/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/Annotations/')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
for xml_file in list_files:
    print(xml_file)
    image = cv2.imread('/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/JPEGImages/' + xml_file[:-4] + '.jpg')
    root = ET.parse('/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/Annotations/' + xml_file).getroot()
    some_objects = root.findall('object')
    for bbox in some_objects:
        label = bbox[0].text
        xmin = int(bbox[4][0].text)
        ymin = int(bbox[4][1].text)
        xmax = int(bbox[4][2].text)
        ymax = int(bbox[4][3].text)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax),(255,0,0),5)
        print(label)
    figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    cv2.imwrite('/home/ubuntu/gfav/data/phase2/frames_with_bbox/' + xml_file[:-4] + '.jpg', image)
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.show()
#     break


# ### Renaming the idlers images and editing the xml's

# In[1]:


import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np


# In[2]:


label_list = os.listdir('/home/ubuntu/gfav/data/phase2/labels/DJI_0002_0/')


# In[3]:


for xml_file in label_list:
    image = cv2.imread('/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/JPEGImages/' + xml_file[:-4] + '.jpg')
    et = ET.parse('/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/Annotations/' + xml_file)
    root = et.getroot()
    file_name = root.findall('filename')
    pathh = root.findall('path')
#     print(pathh[0].text, type(pathh[0].text))
    file_name[0].text = file_name[0].text[4:8] + file_name[0].text[9:10] +         format(int(file_name[0].text[11:-4]), '05d') + file_name[0].text[-4:]
    pathh[0].text = pathh[0].text[:12] + file_name[0].text
    et.write('/home/ubuntu/gfav/data/phase2/labels_int_format/DJI_0002_0/' + file_name[0].text[:-4] + '.xml')
#     print(pathh[0].text, type(pathh[0].text))
    cv2.imwrite("/home/ubuntu/gfav/data/phase2/rgb_frames_int/DJI_0002_0/" + file_name[0].text, image)


# In[2]:


label_list = os.listdir('/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/Annotations/')


# In[3]:


for xml_file in label_list:
    et = ET.parse('/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/Annotations/' + xml_file)
    root = et.getroot()
    some_objects = root.findall('object')
    for bbox in some_objects:
        bbox[2].text = '0'
        bbox[3].text = '0'
    et.write('/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/Annotations/' + xml_file)


# In[ ]:




