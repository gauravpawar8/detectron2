#!/usr/bin/env python
# coding: utf-8

# ### some rough work

# In[1]:


import exiftool


# In[2]:


# files = ["/home/ubuntu/gfav/data/phase2/seq_files/DJI_0004.SEQ"]
# with exiftool.ExifTool() as et:
#     metadata = et.get_metadata_batch(files)


# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
# cv2.VideoWriter_fourcc('X','V', 'I', 'D')
# cap = cv2.VideoCapture('/output/output/DJI_0004jpeg.avi')
cap = cv2.VideoCapture('/home/ubuntu/gfav/data/phase2/thermal_videos/DJI_0003.avi')


# In[21]:


# cap.set('6', cv2.CAP_PROP_FOURCC('M','P', 'N', 'G'))


# In[2]:


i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
#     print(frame.shape)
    break
    if i%30 == 0:
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         plt.imshow(frame)
        plt.show()
#     break
    i += 1
cap.release()


# In[3]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
# cv2.VideoWriter_fourcc('X','V', 'I', 'D')
# cap = cv2.VideoCapture('/output/output/DJI_0004jpeg.avi')
cap = cv2.VideoCapture('/output/output/DJI_0004.avi')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'notebook')
while(cap.isOpened()):
    ret, frame = cap.read(cv2.IMREAD_UNCHANGED)
    if frame is None:
        print('nothing is there')
        break
    plt.imshow(frame)
    break
cap.release()


# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
# cv2.VideoWriter_fourcc('X','V', 'I', 'D')
# cap = cv2.VideoCapture('/output/output/DJI_0004jpeg.avi')
cap = cv2.VideoCapture('/output/output/DJI_0004jpeg.avi')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'notebook')
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        print('nothing is there')
        break
    plt.imshow(frame)
    break
cap.release()


# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
# cv2.VideoWriter_fourcc('X','V', 'I', 'D')
# cap = cv2.VideoCapture('/output/output/DJI_0004jpeg.avi')
cap = cv2.VideoCapture('/output/output/DJI_0004tiff.avi', cv2.IMREAD_UNCHANGED)


# In[2]:


get_ipython().run_line_magic('matplotlib', 'notebook')
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        print('nothing is there')
        break
    plt.imshow(frame)
    break
cap.release()


# In[6]:


pil_image = Image.open('/home/ubuntu/gfav/data/phase2/seq_files/temp/frame00453.tiff')
# im.show()
numpy_array = np.array(pil_image)


# In[9]:


plt.imshow(numpy_array)


# ### read tiff frames

# In[1]:


from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import os


# In[2]:


tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/DJI_3011_0/')
tiff_list.sort()


# In[3]:


rgb_list = os.listdir('/home/ubuntu/gfav/data/phase3/rgb_frames_int/DJI_3011_0/')
rgb_list.sort()
# print(rgb_list)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'notebook')
i = 0
for tiff_image_name in tiff_list:
    if tiff_image_name[-4:] != 'tiff':
        continue
    mat_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/DJI_3009_0/' + tiff_image_name)
    if i%90==0:
        plt.imshow(mat_image)
        plt.show()
    i += 1
    break
    


# ### find range for tiff images

# In[8]:


vmin = 10000
vmax = 0
vminlist = []
vmaxlist = []
dir_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/')
for dir_name in dir_list:
    vmin = 10000
    vmax = 0
    if dir_name[:3] != 'DJI':
        continue
    tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/')
    for tiff_image_name in tiff_list:
        if tiff_image_name[-4:] != 'tiff':
            continue
        tiff_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/' + tiff_image_name)
        vminlist.append(tiff_image.min())
        vmaxlist.append(tiff_image.max())
        if tiff_image.min() < vmin:
            vmin = tiff_image.min()
        if tiff_image.max() > vmax:
            vmax = tiff_image.max()
    print(vmin, vmax, dir_name)


# In[23]:


plt.hist(vminlist, bins=20)


# In[20]:


xt = []
xtt = []
for i in range(2200, 3900, 100):
    xt.append(i)
for i in range(22, 39, 1):
    xtt.append(i)


# In[22]:


plt.hist(vmaxlist, bins=17)
plt.xticks(xt, xtt) 


# In[2]:


vmin = 2000
vmax = 3100


# ### apply color pallette to tiff images

# In[2]:


vmin = 10000
vmax = 0
vminlist = []
vmaxlist = []
dir_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/')
for dir_name in dir_list:
    if dir_name!= 'DJI_3002_0':
        continue
    vmin = 10000
    vmax = 0
    if dir_name[:3] != 'DJI':
        continue
    tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/')
    for tiff_image_name in tiff_list:
        if tiff_image_name[-4:] != 'tiff':
            continue
        tiff_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/' + tiff_image_name)
        vminlist.append(tiff_image.min())
        vmaxlist.append(tiff_image.max())
        if tiff_image.min() < vmin:
            vmin = tiff_image.min()
        if tiff_image.max() > vmax:
            vmax = tiff_image.max()
    print(vmin, vmax, dir_name)


# In[3]:


histt = plt.hist(vminlist, bins=20)
thresh = 0.05 * np.sum(histt[0])
bins = 20
summ = 0
vmin = 0
for i in range(bins):
    summ += histt[0][i]
    if summ >= thresh:
        vmin = int(histt[1][i])
        print('vmin:', vmin)
        break


# In[4]:


histt = plt.hist(vmaxlist, bins=20)
thresh = 0.05 * np.sum(histt[0])
bins = 20
summ = 0
vmax = 0
for i in range(bins-1, -1, -1):
    summ += histt[0][i]
    if summ >= thresh:
        vmax = int(histt[1][i])
        print('vmax:', vmax)
        break


# In[4]:


dir_name = 'DJI_3002_0/'
tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name)
tiff_list.sort()
rgb_list = os.listdir('/home/ubuntu/gfav/data/phase3/rgb_frames_int/' + dir_name)
rgb_list.sort()
# %matplotlib notebook
cmap = plt.cm.jet
for rgb_filename in rgb_list:
    if rgb_filename[-3:] !='jpg':
        continue
    rgb_image = cv2.imread('/home/ubuntu/gfav/data/phase3/rgb_frames_int/' + dir_name + rgb_filename)
    rgb_crop = rgb_image[426:1741, 1001:2776, :].copy()
    ind = int(rgb_filename[-9:-4])
    tiff_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + 'frame' +                             format(int((90*ind + 1)/0.99), '05d') + '.tiff')
    print(tiff_image.min(), tiff_image.max())
    norm = plt.Normalize(vmin, vmax)
    tiff_image = cmap(norm(tiff_image))
#     tiff_image = cmap(tiff_image)
    # save the image
    plt.imsave('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename, tiff_image)
    ir_image = cv2.imread('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename)
    ir_resize = cv2.resize(ir_image, (rgb_crop.shape[1], rgb_crop.shape[0]))
    cv2.imwrite('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename, ir_resize)
    cv2.imwrite('/home/ubuntu/gfav/data/phase3/rgb_frames_crop/' + dir_name + rgb_filename, rgb_crop)
    figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(ir_resize, cv2.COLOR_BGR2RGB))
    plt.show()
    


# In[6]:


vmin = 10000
vmax = 0
vminlist = []
vmaxlist = []
dir_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/')
for dir_name in dir_list:
    if dir_name!= 'DJI_3005_0':
        continue
    vmin = 10000
    vmax = 0
    if dir_name[:3] != 'DJI':
        continue
    tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/')
    for tiff_image_name in tiff_list:
        if tiff_image_name[-4:] != 'tiff':
            continue
        tiff_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/' + tiff_image_name)
        vminlist.append(tiff_image.min())
        vmaxlist.append(tiff_image.max())
        if tiff_image.min() < vmin:
            vmin = tiff_image.min()
        if tiff_image.max() > vmax:
            vmax = tiff_image.max()
    print(vmin, vmax, dir_name)


# In[7]:


histt = plt.hist(vminlist, bins=20)
thresh = 0.05 * np.sum(histt[0])
bins = 20
summ = 0
vmin = 0
for i in range(bins):
    summ += histt[0][i]
    if summ >= thresh:
        vmin = int(histt[1][i])
        print('vmin:', vmin)
        break


# In[8]:


histt = plt.hist(vmaxlist, bins=20)
thresh = 0.05 * np.sum(histt[0])
bins = 20
summ = 0
vmax = 0
for i in range(bins-1, -1, -1):
    summ += histt[0][i]
    if summ >= thresh:
        vmax = int(histt[1][i])
        print('vmax:', vmax)
        break


# In[1]:


# # %matplotlib notebook
# dir_name = 'DJI_3005_0/'
# tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name)
# tiff_list.sort()
# rgb_list = os.listdir('/home/ubuntu/gfav/data/phase3/rgb_frames_int/' + dir_name)
# rgb_list.sort()
# # %matplotlib notebook
# cmap = plt.cm.jet
# for rgb_filename in rgb_list:
#     if rgb_filename[-3:] !='jpg':
#         continue
#     rgb_image = cv2.imread('/home/ubuntu/gfav/data/phase3/rgb_frames_int/' + dir_name + rgb_filename)
#     rgb_crop = rgb_image[426:1741, 1001:2776, :].copy()
#     ind = int(rgb_filename[-9:-4])
#     tiff_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + 'frame' + \
#                             format(int((90*ind + 1)/0.99), '05d') + '.tiff')
#     print(tiff_image.min(), tiff_image.max())
#     norm = plt.Normalize(vmin, vmax)
#     tiff_image = cmap(norm(tiff_image))
# #     tiff_image = cmap(tiff_image)
#     # save the image
#     plt.imsave('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename, tiff_image)
#     ir_image = cv2.imread('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename)
#     ir_resize = cv2.resize(ir_image, (rgb_crop.shape[1], rgb_crop.shape[0]))
#     cv2.imwrite('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename, ir_resize)
#     cv2.imwrite('/home/ubuntu/gfav/data/phase3/rgb_frames_crop/' + dir_name + rgb_filename, rgb_crop)
#     figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
#     plt.subplot(1, 2, 2)
#     plt.imshow(cv2.cvtColor(ir_resize, cv2.COLOR_BGR2RGB))
#     plt.show()
# #     break


# In[10]:


vmin = 10000
vmax = 0
vminlist = []
vmaxlist = []
dir_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/')
for dir_name in dir_list:
    if dir_name!= 'DJI_3007_0':
        continue
    vmin = 10000
    vmax = 0
    if dir_name[:3] != 'DJI':
        continue
    tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/')
    for tiff_image_name in tiff_list:
        if tiff_image_name[-4:] != 'tiff':
            continue
        tiff_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/' + tiff_image_name)
        vminlist.append(tiff_image.min())
        vmaxlist.append(tiff_image.max())
        if tiff_image.min() < vmin:
            vmin = tiff_image.min()
        if tiff_image.max() > vmax:
            vmax = tiff_image.max()
    print(vmin, vmax, dir_name)


# In[11]:


histt = plt.hist(vminlist, bins=20)
thresh = 0.05 * np.sum(histt[0])
bins = 20
summ = 0
vmin = 0
for i in range(bins):
    summ += histt[0][i]
    if summ >= thresh:
        vmin = int(histt[1][i])
        print('vmin:', vmin)
        break


# In[12]:


histt = plt.hist(vmaxlist, bins=20)
thresh = 0.05 * np.sum(histt[0])
bins = 20
summ = 0
vmax = 0
for i in range(bins-1, -1, -1):
    summ += histt[0][i]
    if summ >= thresh:
        vmax = int(histt[1][i])
        print('vmax:', vmax)
        break


# In[2]:


# dir_name = 'DJI_3007_0/'
# tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name)
# tiff_list.sort()
# rgb_list = os.listdir('/home/ubuntu/gfav/data/phase3/rgb_frames_int/' + dir_name)
# rgb_list.sort()
# # %matplotlib notebook
# cmap = plt.cm.jet
# for rgb_filename in rgb_list:
#     if rgb_filename[-3:] !='jpg':
#         continue
#     rgb_image = cv2.imread('/home/ubuntu/gfav/data/phase3/rgb_frames_int/' + dir_name + rgb_filename)
#     rgb_crop = rgb_image[426:1741, 1001:2776, :].copy()
#     ind = int(rgb_filename[-9:-4])
#     tiff_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + 'frame' + \
#                             format(int((90*ind + 1)/0.99), '05d') + '.tiff')
#     print(tiff_image.min(), tiff_image.max())
#     norm = plt.Normalize(vmin, vmax)
#     tiff_image = cmap(norm(tiff_image))
# #     tiff_image = cmap(tiff_image)
#     # save the image
#     plt.imsave('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename, tiff_image)
#     ir_image = cv2.imread('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename)
#     ir_resize = cv2.resize(ir_image, (rgb_crop.shape[1], rgb_crop.shape[0]))
#     cv2.imwrite('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename, ir_resize)
#     cv2.imwrite('/home/ubuntu/gfav/data/phase3/rgb_frames_crop/' + dir_name + rgb_filename, rgb_crop)
#     figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
#     plt.subplot(1, 2, 2)
#     plt.imshow(cv2.cvtColor(ir_resize, cv2.COLOR_BGR2RGB))
#     plt.show()
    


# In[14]:


vmin = 10000
vmax = 0
vminlist = []
vmaxlist = []
dir_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/')
for dir_name in dir_list:
    if dir_name!= 'DJI_3009_0':
        continue
    vmin = 10000
    vmax = 0
    if dir_name[:3] != 'DJI':
        continue
    tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/')
    for tiff_image_name in tiff_list:
        if tiff_image_name[-4:] != 'tiff':
            continue
        tiff_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/' + tiff_image_name)
        vminlist.append(tiff_image.min())
        vmaxlist.append(tiff_image.max())
        if tiff_image.min() < vmin:
            vmin = tiff_image.min()
        if tiff_image.max() > vmax:
            vmax = tiff_image.max()
    print(vmin, vmax, dir_name)


# In[15]:


histt = plt.hist(vminlist, bins=20)
thresh = 0.05 * np.sum(histt[0])
bins = 20
summ = 0
vmin = 0
for i in range(bins):
    summ += histt[0][i]
    if summ >= thresh:
        vmin = int(histt[1][i])
        print('vmin:', vmin)
        break


# In[16]:


histt = plt.hist(vmaxlist, bins=20)
thresh = 0.05 * np.sum(histt[0])
bins = 20
summ = 0
vmax = 0
for i in range(bins-1, -1, -1):
    summ += histt[0][i]
    if summ >= thresh:
        vmax = int(histt[1][i])
        print('vmax:', vmax)
        break


# In[3]:


# dir_name = 'DJI_3009_0/'
# tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name)
# tiff_list.sort()
# rgb_list = os.listdir('/home/ubuntu/gfav/data/phase3/rgb_frames_int/' + dir_name)
# rgb_list.sort()
# # %matplotlib notebook
# cmap = plt.cm.jet
# for rgb_filename in rgb_list:
#     if rgb_filename[-3:] !='jpg':
#         continue
#     rgb_image = cv2.imread('/home/ubuntu/gfav/data/phase3/rgb_frames_int/' + dir_name + rgb_filename)
#     rgb_crop = rgb_image[426:1741, 1001:2776, :].copy()
#     ind = int(rgb_filename[-9:-4])
#     tiff_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + 'frame' + \
#                             format(int((90*ind + 1)/0.99), '05d') + '.tiff')
#     print(tiff_image.min(), tiff_image.max())
#     norm = plt.Normalize(vmin, vmax)
#     tiff_image = cmap(norm(tiff_image))
# #     tiff_image = cmap(tiff_image)
#     # save the image
#     plt.imsave('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename, tiff_image)
#     ir_image = cv2.imread('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename)
#     ir_resize = cv2.resize(ir_image, (rgb_crop.shape[1], rgb_crop.shape[0]))
#     cv2.imwrite('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename, ir_resize)
#     cv2.imwrite('/home/ubuntu/gfav/data/phase3/rgb_frames_crop/' + dir_name + rgb_filename, rgb_crop)
#     figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
#     plt.subplot(1, 2, 2)
#     plt.imshow(cv2.cvtColor(ir_resize, cv2.COLOR_BGR2RGB))
#     plt.show()
    


# In[18]:


vmin = 10000
vmax = 0
vminlist = []
vmaxlist = []
dir_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/')
for dir_name in dir_list:
    if dir_name!= 'DJI_3010_0':
        continue
    vmin = 10000
    vmax = 0
    if dir_name[:3] != 'DJI':
        continue
    tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/')
    for tiff_image_name in tiff_list:
        if tiff_image_name[-4:] != 'tiff':
            continue
        tiff_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/' + tiff_image_name)
        vminlist.append(tiff_image.min())
        vmaxlist.append(tiff_image.max())
        if tiff_image.min() < vmin:
            vmin = tiff_image.min()
        if tiff_image.max() > vmax:
            vmax = tiff_image.max()
    print(vmin, vmax, dir_name)


# In[19]:


histt = plt.hist(vminlist, bins=20)
thresh = 0.05 * np.sum(histt[0])
bins = 20
summ = 0
vmin = 0
for i in range(bins):
    summ += histt[0][i]
    if summ >= thresh:
        vmin = int(histt[1][i])
        print('vmin:', vmin)
        break


# In[20]:


histt = plt.hist(vmaxlist, bins=20)
thresh = 0.05 * np.sum(histt[0])
bins = 20
summ = 0
vmax = 0
for i in range(bins-1, -1, -1):
    summ += histt[0][i]
    if summ >= thresh:
        vmax = int(histt[1][i])
        print('vmax:', vmax)
        break


# In[27]:


# dir_name = 'DJI_3010_0/'
# tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name)
# tiff_list.sort()
# rgb_list = os.listdir('/home/ubuntu/gfav/data/phase3/rgb_frames_int/' + dir_name)
# rgb_list.sort()
# # %matplotlib notebook
# cmap = plt.cm.jet
# for rgb_filename in rgb_list:
#     if rgb_filename[-3:] !='jpg':
#         continue
#     rgb_image = cv2.imread('/home/ubuntu/gfav/data/phase3/rgb_frames_int/' + dir_name + rgb_filename)
#     rgb_crop = rgb_image[426:1741, 1001:2776, :].copy()
#     ind = int(rgb_filename[-9:-4])
#     tiff_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + 'frame' + \
#                             format(int((90*ind + 1)/0.99), '05d') + '.tiff')
#     print(tiff_image.min(), tiff_image.max())
#     norm = plt.Normalize(vmin, vmax)
#     tiff_image = cmap(norm(tiff_image))
# #     tiff_image = cmap(tiff_image)
#     # save the image
#     plt.imsave('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename, tiff_image)
#     ir_image = cv2.imread('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename)
#     ir_resize = cv2.resize(ir_image, (rgb_crop.shape[1], rgb_crop.shape[0]))
#     cv2.imwrite('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename, ir_resize)
#     cv2.imwrite('/home/ubuntu/gfav/data/phase3/rgb_frames_crop/' + dir_name + rgb_filename, rgb_crop)
#     figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
#     plt.subplot(1, 2, 2)
#     plt.imshow(cv2.cvtColor(ir_resize, cv2.COLOR_BGR2RGB))
#     plt.show()
    


# In[22]:


vmin = 10000
vmax = 0
vminlist = []
vmaxlist = []
dir_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/')
for dir_name in dir_list:
    if dir_name!= 'DJI_3011_0':
        continue
    vmin = 10000
    vmax = 0
    if dir_name[:3] != 'DJI':
        continue
    tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/')
    for tiff_image_name in tiff_list:
        if tiff_image_name[-4:] != 'tiff':
            continue
        tiff_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + '/' + tiff_image_name)
        vminlist.append(tiff_image.min())
        vmaxlist.append(tiff_image.max())
        if tiff_image.min() < vmin:
            vmin = tiff_image.min()
        if tiff_image.max() > vmax:
            vmax = tiff_image.max()
    print(vmin, vmax, dir_name)


# In[23]:


histt = plt.hist(vminlist, bins=20)
thresh = 0.05 * np.sum(histt[0])
bins = 20
summ = 0
vmin = 0
for i in range(bins):
    summ += histt[0][i]
    if summ >= thresh:
        vmin = int(histt[1][i])
        print('vmin:', vmin)
        break


# In[24]:


histt = plt.hist(vmaxlist, bins=20)
thresh = 0.05 * np.sum(histt[0])
bins = 20
summ = 0
vmax = 0
for i in range(bins-1, -1, -1):
    summ += histt[0][i]
    if summ >= thresh:
        vmax = int(histt[1][i])
        print('vmax:', vmax)
        break


# In[26]:


# dir_name = 'DJI_3011_0/'
# tiff_list = os.listdir('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name)
# tiff_list.sort()
# rgb_list = os.listdir('/home/ubuntu/gfav/data/phase3/rgb_frames_int/' + dir_name)
# rgb_list.sort()
# # %matplotlib notebook
# cmap = plt.cm.jet
# for rgb_filename in rgb_list:
#     if rgb_filename[-3:] !='jpg':
#         continue
#     rgb_image = cv2.imread('/home/ubuntu/gfav/data/phase3/rgb_frames_int/' + dir_name + rgb_filename)
#     rgb_crop = rgb_image[426:1741, 1001:2776, :].copy()
#     ind = int(rgb_filename[-9:-4])
#     tiff_image = plt.imread('/home/ubuntu/gfav/data/phase3/tiff_frames/' + dir_name + 'frame' + \
#                             format(int((90*ind + 1)/0.99), '05d') + '.tiff')
#     print(tiff_image.min(), tiff_image.max())
#     norm = plt.Normalize(vmin, vmax)
#     tiff_image = cmap(norm(tiff_image))
# #     tiff_image = cmap(tiff_image)
#     # save the image
#     plt.imsave('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename, tiff_image)
#     ir_image = cv2.imread('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename)
#     ir_resize = cv2.resize(ir_image, (rgb_crop.shape[1], rgb_crop.shape[0]))
#     cv2.imwrite('/home/ubuntu/gfav/data/phase3/ir_frames/' + dir_name + rgb_filename, ir_resize)
#     cv2.imwrite('/home/ubuntu/gfav/data/phase3/rgb_frames_crop/' + dir_name + rgb_filename, rgb_crop)
#     figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
#     plt.subplot(1, 2, 2)
#     plt.imshow(cv2.cvtColor(ir_resize, cv2.COLOR_BGR2RGB))
#     plt.show()
    


# In[ ]:




