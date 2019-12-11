#!/usr/bin/env python
# coding: utf-8

# ## Detectron2 beginner tutorial

# ### Some basic setup

# In[1]:


# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
import os
from matplotlib.pyplot import figure
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# ### Inference with pre-trained models

# In[2]:


# !wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg


# In[3]:


im = cv2.imread("./input.jpg")


# In[4]:


plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


# In[5]:


cfg = get_cfg()
cfg.merge_from_file("/home/ubuntu/gfav/repos/detectron2/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml")
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
cfg.MODEL.WEIGHTS = "/home/ubuntu/gfav/repos/detectron2/pretrained_models/RetinaNet/model_final_59f53c.pkl"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)


# In[6]:


# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
outputs["instances"].pred_classes 
outputs["instances"].pred_boxes 
outputs["instances"].scores 


# In[7]:


# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))


# In[8]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[9]:


plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))


# In[10]:


outputs


# In[12]:


cfg


# ### Training RetinaNet 

# In[1]:


# python tools/train_net.py --config-file configs/PascalVOC-Detection/retinanet_R_101_FPN_3x.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.01


# ### Inference with RetinaNet trained on PascalVOC

# In[17]:


im = cv2.imread("/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/JPEGImages/002596.jpg")
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


# In[18]:


cfg = get_cfg()
cfg.merge_from_file("/home/ubuntu/gfav/repos/detectron2/configs/PascalVOC-Detection/retinanet_R_101_FPN_3x.yaml")
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
cfg.MODEL.WEIGHTS = "/home/ubuntu/gfav/repos/detectron2/output_pascalvoc_exp2/model_final.pth"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)


# In[7]:


cfg


# In[19]:


outputs


# In[20]:


# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))


# In[21]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))


# ### Inference with RetinaNet trained on idler data

# In[2]:


cfg = get_cfg()
cfg.merge_from_file("/home/ubuntu/gfav/repos/detectron2/configs/PascalVOC-Detection/retinanet_R_101_FPN_3x.yaml")
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3  # set threshold for this model
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
cfg.MODEL.WEIGHTS = "/home/ubuntu/gfav/repos/detectron2/output_idler_exp15/model_0004799.pth"
predictor = DefaultPredictor(cfg)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
with open('/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/ImageSets/Main/test_exp4.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content] 
for im_name in content:
    im = cv2.imread("/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/JPEGImages/" + im_name + '.jpg')
    outputs = predictor(im)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    cv2.imwrite('/home/ubuntu/gfav/data/phase2/exp4_results_without_thresh/' + im_name + '.jpg', v.get_image()[:, :, ::-1])
#     plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#     plt.show()
#     break


# ## extract IR and RGB crops

# In[19]:


# img_list = os.listdir('/home/ubuntu/gfav/data/phase3/shortlisted_frames/')
# for img_name in img_list:
#     if img_name[-3:] == 'png':
#         ir_image = cv2.imread('/home/ubuntu/gfav/data/phase3/shortlisted_frames/' + img_name)
#         ir_crop = ir_image[3:668, 220:1085, :].copy()
#         cv2.imwrite('/home/ubuntu/gfav/data/phase3/shortlisted_frames/cropped_IR/' + img_name, ir_crop)
# #         figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
# #         plt.imshow(cv2.cvtColor(ir_crop, cv2.COLOR_BGR2RGB))
# #         plt.show()
#     if img_name[-3:] == 'jpg':
#         rgb_image = cv2.imread('/home/ubuntu/gfav/data/phase3/shortlisted_frames/' + img_name)
#         rgb_crop = rgb_image[440:1640, 1040:2750, :].copy()
#         cv2.imwrite('/home/ubuntu/gfav/data/phase3/shortlisted_frames/cropped_RGB/' + img_name, rgb_crop)
# #         figure(num=None, figsize=(4, 3), dpi=200, facecolor='w', edgecolor='k')
# #         plt.imshow(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
# #         plt.show()


# In[18]:


# %matplotlib notebook
# img_list = os.listdir('/home/ubuntu/gfav/data/phase3/shortlisted_frames/output_RGB/')
# for img_name in img_list:
    
#     ir_image = cv2.imread('/home/ubuntu/gfav/data/phase3/shortlisted_frames/' + img_name[:-3] + 'png')
#     ir_crop = ir_image[3:668, 220:1085, :].copy()
# #         cv2.imwrite('/home/ubuntu/gfav/data/phase3/shortlisted_frames/cropped_IR/' + img_name, ir_crop)
#     figure(num=None, figsize=(4, 3), dpi=200, facecolor='w', edgecolor='k')
#     plt.subplot(1,2,1)
#     plt.imshow(cv2.cvtColor(ir_crop, cv2.COLOR_BGR2RGB))

#     rgb_image = cv2.imread('/home/ubuntu/gfav/data/phase3/shortlisted_frames/' + img_name)
#     rgb_crop = rgb_image[440:1640, 1040:2750, :].copy()
#     rgb_crop = cv2.resize(rgb_crop, (ir_crop.shape[1], ir_crop.shape[0]))
# #         cv2.imwrite('/home/ubuntu/gfav/data/phase3/shortlisted_frames/cropped_RGB/' + img_name, rgb_crop)
#     plt.subplot(1,2,2)
#     plt.imshow(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
#     plt.show()
#     break


# In[6]:


# %matplotlib notebook
# ir_image = cv2.imread('/home/ubuntu/gfav/data/phase3/ir_frames/DJI_3009_0/3009000000.jpg')
# print('ir_image shape:', ir_image.shape)
# rgb_image = cv2.imread('/home/ubuntu/gfav/data/phase3/rgb_frames_int/DJI_3009_0/3009000000.jpg')
# print('rgb_image shape:', rgb_image.shape)
# rgb_crop = rgb_image[413:1753, 1050:2770, :].copy()
# print('rgb_crop shape:', rgb_crop.shape)
# ir_resize = cv2.resize(ir_image, (rgb_crop.shape[1], rgb_crop.shape[0]))
# figure(num=None, figsize=(4, 3), dpi=200, facecolor='w', edgecolor='k')
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(ir_image, cv2.COLOR_BGR2RGB))
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
# plt.show()
# figure(num=None, figsize=(4, 3), dpi=200, facecolor='w', edgecolor='k')
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(ir_image, cv2.COLOR_BGR2RGB))
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
# plt.show()
# figure(num=None, figsize=(4, 3), dpi=200, facecolor='w', edgecolor='k')
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(ir_resize, cv2.COLOR_BGR2RGB))
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
# plt.show()


# ### map ir and rgb image

# In[5]:


# frame_name: 3005000000.jpg
# (x1, y1)
ir_1 = (113, 158)
rgb_1 = (1598, 1238)
ir_2 = (279, 128)
rgb_2 = (2475, 1084)
ir_to_rgb_x_axis = (ir_2[0] - ir_1[0]) / (rgb_2[0] - rgb_1[0])
ir_to_rgb_y_axis = (ir_1[1] - ir_2[1]) / (rgb_1[1] - rgb_2[1])
ir_shape = (256, 336) # (row, col) or (y, x)
left_margin_rgb = ir_1[0] / ir_to_rgb_x_axis
right_margin_rgb = (ir_shape[1] - ir_2[0]) / ir_to_rgb_x_axis
left_cord_rgb = int(rgb_1[0] - left_margin_rgb)
right_cord_rgb = int(rgb_2[0] + right_margin_rgb)
print(left_cord_rgb, right_cord_rgb)
top_margin_rgb = ir_2[1] / ir_to_rgb_y_axis
bottom_margin_rgb = (ir_shape[0] - ir_1[1]) / ir_to_rgb_y_axis
top_cord_rgb = int(rgb_2[1] - top_margin_rgb)
bottom_cord_rgb = int(rgb_1[1] + bottom_margin_rgb)
print(top_cord_rgb, bottom_cord_rgb)


# In[8]:


get_ipython().run_line_magic('matplotlib', 'notebook')
ir_image = cv2.imread('/home/ubuntu/gfav/data/phase3/ir_frames/DJI_3005_0/3005000000.jpg')
print('ir_image shape:', ir_image.shape)
rgb_image = cv2.imread('/home/ubuntu/gfav/data/phase3/rgb_frames_int/DJI_3005_0/3005000000.jpg')
print('rgb_image shape:', rgb_image.shape)
rgb_crop = rgb_image[426:1741, 1001:2776, :].copy()
print('rgb_crop shape:', rgb_crop.shape)
ir_resize = cv2.resize(ir_image, (rgb_crop.shape[1], rgb_crop.shape[0]))
figure(num=None, figsize=(4, 3), dpi=200, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(ir_image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
plt.show()
figure(num=None, figsize=(4, 3), dpi=200, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(ir_image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
plt.show()
figure(num=None, figsize=(4, 3), dpi=200, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(ir_resize, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
plt.show()


# ### apply retinanet on IR and RGB crops

# In[10]:


dir_name = 'DJI_3011_0/'
img_list = os.listdir('/home/ubuntu/gfav/data/phase3/rgb_frames_crop/' + dir_name)
for img_file in img_list:
    img = cv2.imread('/home/ubuntu/gfav/data/phase3/rgb_frames_crop/' + dir_name + img_file)
    if img is None:
        continue
    ir_img = cv2.imread('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + img_file)
    ir_copy = ir_img.copy()
    im = img.copy()
    outputs = predictor(im)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
#     plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#     plt.show()
    cv2.imwrite('/home/ubuntu/gfav/data/phase3/rgb_output/' + dir_name + img_file, v.get_image()[:, :, ::-1])
    v = Visualizer(ir_copy[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    cv2.imwrite('/home/ubuntu/gfav/data/phase3/ir_output/' + dir_name + img_file, v.get_image()[:, :, ::-1])
#     plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#     plt.show()


# In[3]:


# %matplotlib notebook
# rgb_image = cv2.imread('/home/ubuntu/gfav/data/phase3/shortlisted_frames/cropped_RGB/DJI_0009_0009.jpg')
# ir_image = cv2.imread('/home/ubuntu/gfav/data/phase3/shortlisted_frames/cropped_IR/DJI_0009_0009.png')
# plt.subplot(2, 1, 1)
# plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
# plt.subplot(2, 1, 2)
# plt.imshow(cv2.cvtColor(ir_image, cv2.COLOR_BGR2RGB))
# plt.show()


# In[4]:


rgb_image.shape


# In[5]:


ir_image.shape


# ### map temperature

# In[3]:


from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET


# In[14]:


temperature_scale_img = cv2.imread('/home/ubuntu/gfav/data/jet_colormap.png')
temperature_scale_img = temperature_scale_img[:, 10:20, :]
atm_temp = 9
resized_scale = cv2.resize(temperature_scale_img, (temperature_scale_img.shape[1], atm_temp))


# In[15]:


# %matplotlib notebook
figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
plt.imshow(cv2.cvtColor(temperature_scale_img, cv2.COLOR_BGR2RGB))
plt.show()


# In[16]:


# %matplotlib notebook
figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
plt.imshow(cv2.cvtColor(resized_scale, cv2.COLOR_BGR2RGB))
plt.show()


# ### finding temperature distribution

# In[17]:


def temperature_distribution(frame, temperature_scale, atm_temp):
    temperature_array = np.ndarray(frame.shape[:2])
    for row in range(frame.shape[0]):
        for col in range(frame.shape[1]):
            bgr_pixel = frame[row, col, :].astype(int)
            diffence_values = []
            for scale_value in temperature_scale:
                diffence_values.append(np.sum(np.abs(scale_value.astype(int) - bgr_pixel)))
            temperature_array[row, col] = atm_temp - np.argmin(diffence_values)
    return temperature_array.astype(int)
#     return bin_values


# In[16]:


# %matplotlib inline
# xml_list = os.listdir('/home/ubuntu/gfav/data/phase3/shortlisted_frames/labels_cropped_ir_manually/')
# for xml_file in xml_list:
#     root = ET.parse('/home/ubuntu/gfav/data/phase3/shortlisted_frames/labels_cropped_ir_manually/' + xml_file).getroot()
#     ir_image = cv2.imread('/home/ubuntu/gfav/data/phase3/shortlisted_frames/cropped_IR/' + xml_file[:-3] + 'png')
#     label_image = ir_image.copy()
#     some_objects = root.findall('object')
#     for bbox in some_objects:
#         xmin = int(bbox[4][0].text)
#         ymin = int(bbox[4][1].text)
#         xmax = int(bbox[4][2].text)
#         ymax = int(bbox[4][3].text)
#         image_crop = ir_image[ymin:ymax, xmin:xmax, :].copy()
#         cv2.rectangle(label_image, (xmin, ymin), (xmax, ymax),(255,0,0),5)
#         temperature_array = temperature_distribution(image_crop, resized_scale[:, 5, :])
#         max_temperature = np.max(temperature_array)
#         (weights, bin_values) = np.histogram(temperature_array, bins=range(-2, 10))
#         cv2.putText(label_image, str(max_temperature - 2), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 4)
#         figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
# #         plt.subplot(2,1,1)
# #         plt.imshow(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
# #         plt.subplot(2,1,2)
# #         plt.hist(temperature_array.flatten(), bins=bin_values, edgecolor='black', log=True)
# #         plt.show()
#         print('max_temperature:', max_temperature - 2)
#         print('weights:', weights)
#         print('max_temperature weight:', weights[max_temperature - 13])
# #         break
#     cv2.imwrite('/home/ubuntu/gfav/data/phase3/shortlisted_frames/output_IR_manual/' + xml_file[:-3] + 'png', label_image)
# #     figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
# #     plt.imshow(cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB))
# #     plt.show
# #     break


# In[21]:


dir_name = 'DJI_3011_0/'
img_list = os.listdir('/home/ubuntu/gfav/data/phase3/rgb_frames_crop/' + dir_name)
for img_file in img_list:
    img = cv2.imread('/home/ubuntu/gfav/data/phase3/rgb_frames_crop/' + dir_name + img_file)
    if img is None:
        continue
    ir_img = cv2.imread('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + img_file)
    ir_copy = ir_img.copy()
    im = img.copy()
    outputs = predictor(im)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
#     plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#     plt.show()
    cv2.imwrite('/home/ubuntu/gfav/data/phase3/rgb_output/' + dir_name + img_file, v.get_image()[:, :, ::-1])
    for bbox_tensor in outputs["instances"].pred_boxes:
        bbox = bbox_tensor.cpu().numpy()
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        image_crop = ir_img[ymin:ymax, xmin:xmax, :].copy()
        image_crop = cv2.resize(image_crop, (0, 0), fx=0.25, fy=0.25)
        temperature_array = temperature_distribution(image_crop, resized_scale[:, 5, :], atm_temp)
        max_temperature = np.max(temperature_array)
#         (weights, bin_values) = np.histogram(temperature_array, bins=range(0, 20))
        cv2.putText(ir_copy, str(max_temperature), (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,205,12), 4)
    cv2.putText(ir_copy, 'Atmospheric temperature:' + str(atm_temp), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,205,12), 4)
    v = Visualizer(ir_copy[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
    cv2.imwrite('/home/ubuntu/gfav/data/phase3/ir_output/' + dir_name + img_file, v.get_image()[:, :, ::-1])
#     plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#     plt.show()
#     break


# In[ ]:




