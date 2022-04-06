import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

# id = '1433581'
# img = sitk.ReadImage('nii/'+id+'.nii')
# img = sitk.GetArrayFromImage(img)
# mask = np.load('lung_mask/'+id+'lungmask.npy')

# print(img.shape)
# print(mask.shape)

# img = img[99+86,:,:]
# mask = mask[99,:,:]
# plt.imshow(mask*img, cmap='gray')
# plt.savefig('1474381mask2.jpg')

# for i in img:
#     i = sitk.GetArrayFromImage(i)
#     plt.imshow(i, cmap = 'gray')
#     plt.savefig('1106619_1.jpg')
#     break
# img = img[:, :, 126:503]
# img = sitk.GetArrayFromImage(img)

# img = img[-1, :, :]
# plt.imshow(img, cmap = 'gray')
# plt.savefig('1106619_2.jpg')

# img = np.load('normalized1/segmented/1106619.npy')
# node_mask = np.load('resampled1/mask/1106619mask.npy')
# lung_mask = np.load('resampled1/lung_mask/1106619lungmask.npy')
# # mask = np.load('/home/lyj_11921026/liuqinxian/NAT/NAT/datasets/processed/lung_mask_clean/1106619.npy')

# img = img[180, :, :]
# node_mask = node_mask[180, :, :]
# lung_mask = lung_mask[180, :, :]
# fig,ax = plt.subplots(2,2,figsize=[10,10])
# ax[0,0].imshow(img)  # CT切片图
# ax[0,1].imshow(img,cmap='gray')  # CT切片灰度图
# ax[1,0].imshow(img*node_mask,cmap='gray')  # 标注mask，标注区域为1，其他为0
# ax[1,1].imshow(img*lung_mask,cmap='gray')  # 标注mask区域切片图
# plt.savefig('1106619_1.jpg')


# img2 = np.load('normalized2/segmented/1106619.npy')
# node_mask = np.load('resampled2/mask/1106619mask.npy')
# lung_mask = np.load('resampled2/lung_mask/1106619lungmask.npy')
# # mask = np.load('/home/lyj_11921026/liuqinxian/NAT/NAT/datasets/processed/lung_mask_clean/1106619.npy')

# img2 = img2[180, :, :]
# node_mask = node_mask[180, :, :]
# lung_mask = lung_mask[180, :, :]
# fig,ax = plt.subplots(2,2,figsize=[10,10])
# ax[0,0].imshow(img2)  # CT切片图
# ax[0,1].imshow(img2,cmap='gray')  # CT切片灰度图
# ax[1,0].imshow(img2*node_mask,cmap='gray')  # 标注mask，标注区域为1，其他为0
# ax[1,1].imshow(img2*lung_mask,cmap='gray')  # 标注mask区域切片图
# plt.savefig('1106619_2.jpg')

img = np.load('cube/img_segmented/1106619.npy')
img = img[1]
img = img[32, :, :]

node_mask = np.load('cube/node_mask/1106619nodemask.npy')
node_mask = node_mask[1]
node_mask = node_mask[32, :, :]
fig, ax = plt.subplots(2, figsize=[10, 10])
ax[0].imshow(img)
ax[1].imshow(node_mask)
plt.savefig('1106619_cube.jpg')
