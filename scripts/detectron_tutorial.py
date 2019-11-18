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

# In[3]:


cfg = get_cfg()
cfg.merge_from_file("/home/ubuntu/gfav/repos/detectron2/configs/PascalVOC-Detection/retinanet_R_101_FPN_3x.yaml")
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
cfg.MODEL.WEIGHTS = "/home/ubuntu/gfav/repos/detectron2/output_idler_exp2/model_final.pth"
predictor = DefaultPredictor(cfg)


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
with open('/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/ImageSets/Main/test.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content] 
for im_name in content:
    im = cv2.imread("/home/ubuntu/gfav/repos/detectron2/datasets/VOC2007/JPEGImages/" + im_name + '.jpg')
    outputs = predictor(im)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
#     cv2.imwrite('/home/ubuntu/gfav/data/phase2/exp1_results/' + im_name + '.jpg', v.get_image()[:, :, ::-1])
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()
    break


# In[ ]:




