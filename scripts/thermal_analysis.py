#!/usr/bin/env python
# coding: utf-8

# ### map temperature

# In[1]:


from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET


# In[2]:


temperature_scale_img = cv2.imread('/home/ubuntu/gfav/data/phase1/IR_data/IR_frames/temperature_scale.png')
temperature_scale_img = temperature_scale_img[:, 25:40, :]
resized_scale = cv2.resize(temperature_scale_img, (temperature_scale_img.shape[1], 40))


# In[3]:


# %matplotlib notebook
figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
plt.imshow(cv2.cvtColor(temperature_scale_img, cv2.COLOR_BGR2RGB))
plt.show()


# In[4]:


# %matplotlib notebook
figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
plt.imshow(cv2.cvtColor(resized_scale, cv2.COLOR_BGR2RGB))
plt.show()


# In[5]:


print(resized_scale[:, 5, 0])
print(resized_scale[:, 5, 1])
print(resized_scale[:, 5, 2])
print(resized_scale[:, 5, :])


# ### finding temperature distribution

# In[6]:


def temperature_distribution(frame, temperature_scale):
    temperature_array = np.ndarray(frame.shape[:2])
    for row in range(frame.shape[0]):
        for col in range(frame.shape[1]):
            bgr_pixel = frame[row, col, :].astype(int)
            diffence_values = []
            for scale_value in temperature_scale:
                diffence_values.append(np.sum(np.abs(scale_value.astype(int) - bgr_pixel)))
            temperature_array[row, col] = 50 - np.argmin(diffence_values)
    return temperature_array.astype(int)
#     return bin_values


# ### read, parse xml and corresponding image

# In[7]:


# image_name = '/home/ubuntu/gfav/data/phase1/IR_data/IR_frames/LangesBand_Flug2_0415_IR.png'
# xml_file = '/home/ubuntu/gfav/data/phase1/IR_data/labeled_xmls/LangesBand_Flug2_0415_IR.xml'
# root = ET.parse(xml_file).getroot()
# ir_image = cv2.imread(image_name)


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
xml_list = os.listdir('/home/ubuntu/gfav/data/phase1/IR_data/labeled_xmls/')
for xml_file in xml_list:
    root = ET.parse('/home/ubuntu/gfav/data/phase1/IR_data/labeled_xmls/' + xml_file).getroot()
    ir_image = cv2.imread('/home/ubuntu/gfav/data/phase1/IR_data/IR_frames/' + xml_file[:-3] + 'png')
    label_image = ir_image.copy()
    some_objects = root.findall('object')
    for bbox in some_objects:
        xmin = int(bbox[4][0].text)
        ymin = int(bbox[4][1].text)
        xmax = int(bbox[4][2].text)
        ymax = int(bbox[4][3].text)
        image_crop = ir_image[ymin:ymax, xmin:xmax, :].copy()
        cv2.rectangle(label_image, (xmin, ymin), (xmax, ymax),(255,0,0),5)
        temperature_array = temperature_distribution(image_crop, resized_scale[:, 5, :])
        max_temperature = np.max(temperature_array)
        (weights, bin_values) = np.histogram(temperature_array, bins=range(11,50))
        cv2.putText(label_image, str(max_temperature), (xmax, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 4)
        figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
        plt.subplot(2,1,1)
        plt.imshow(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        plt.subplot(2,1,2)
        plt.hist(temperature_array.flatten(), bins=bin_values, edgecolor='black', log=True)
        plt.show()
        print('max_temperature:', max_temperature)
        print('weights:', weights)
        print('max_temperature weight:', weights[max_temperature - 11])
#         break
    figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    plt.imshow(cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB))
    plt.show
    break


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
xml_list = os.listdir('/home/ubuntu/gfav/data/phase1/IR_data/labeled_xmls/')
for xml_file in xml_list:
    root = ET.parse('/home/ubuntu/gfav/data/phase1/IR_data/labeled_xmls/' + xml_file).getroot()
    ir_image = cv2.imread('/home/ubuntu/gfav/data/phase1/IR_data/IR_frames/' + xml_file[:-3] + 'png')
    label_image = ir_image.copy()
    some_objects = root.findall('object')
    for bbox in some_objects:
        xmin = int(bbox[4][0].text)
        ymin = int(bbox[4][1].text)
        xmax = int(bbox[4][2].text)
        ymax = int(bbox[4][3].text)
        image_crop = ir_image[ymin:ymax, xmin:xmax, :].copy()
        cv2.rectangle(label_image, (xmin, ymin), (xmax, ymax),(255,0,0),5)
        temperature_array = temperature_distribution(image_crop, resized_scale[:, 5, :])
        max_temperature = np.max(temperature_array)
        (weights, bin_values) = np.histogram(temperature_array, bins=range(11,50))
        cv2.putText(label_image, str(max_temperature), (xmax, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 4)
        figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
        plt.subplot(2,1,1)
        plt.imshow(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        plt.subplot(2,1,2)
        plt.hist(temperature_array.flatten(), bins=bin_values, edgecolor='black', log=True)
        plt.show()
        print('max_temperature:', max_temperature)
        print('weights:', weights)
        print('max_temperature weight:', weights[max_temperature - 11])
#         break
    figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    plt.imshow(cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB))
    plt.show
#     break

