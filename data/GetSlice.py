import numpy as np
import pandas as pd

step_size = 3
n_slice = 5
total_size = 256

df = pd.read_csv('labels.txt', sep='\t')
df.sort_values(by='ID', ascending=True)

# 获得所有ID
ids = np.array(df['ID'])

for id in ids:
    id = str(id)
    mask = np.load('resampled/node_mask/'+id+'nodemask.npy')
    img = np.load('normalized/img_segmented/'+id+'.npy')
    
    # 计算肿瘤中心
    z, y, x = np.where(mask>0)
    z, y, x = (np.max(z)+np.min(z))//2, (np.max(y)+np.min(y))//2, (np.max(x)+np.min(x))//2
    
    # 将图片补齐避免超出范围
    z_size, y_size, x_size = img.shape
    pad_img = np.zeros((z_size, y_size+total_size, x_size+total_size))
    pad_img[:, total_size//2 : total_size//2+y_size, total_size//2 : total_size//2+x_size] = img
    
    # 从中心点开始以固定步长采样图片
    slices = []
    for i in range(n_slice):
        idx = z + (i - n_slice // 2) * step_size
        slice = pad_img[idx, y : y+total_size, x : x+total_size]
        slices.append(slice)
    slices = np.stack(slices)
    print(slices.shape)
    np.save('slice/' + str(step_size) + '_' + str(n_slice) + '/img_segmented/'+id+'.npy', slices)
    
    z_size, y_size, x_size = mask.shape
    pad_mask = np.zeros((z_size, y_size+total_size, x_size+total_size))
    pad_mask[:, total_size//2 : total_size//2+y_size, total_size//2 : total_size//2+x_size] = mask
    slices = []
    for i in range(n_slice):
        idx = z + (i - n_slice // 2) * step_size
        slice = pad_mask[idx, y : y+total_size, x : x+total_size]
        slices.append(slice)
    slices = np.stack(slices)
    print(slices.shape)
    np.save('slice/' + str(step_size) + '_' + str(n_slice) + '/node_mask/'+id+'nodemask.npy', slices)