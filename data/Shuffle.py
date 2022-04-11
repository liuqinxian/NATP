import pandas as pd
import numpy as np
import random

# 随机种子设置

df = pd.read_csv('labels.txt', sep='\t')
df.set_index('ID', inplace=True)

# 获得所有ID并打乱
ids = np.array(df.index)
random.shuffle(ids)

imgs = []
masks = []
labels = []
for id in ids:
    label = df.loc[id, 'MPR']
    labels.append(label)

    id = str(id)
    img = np.load('cube/img_segmented/'+id+'.npy')
    imgs.append(img)
    
    mask = np.load('cube/node_mask/'+id+'nodemask.npy')
    masks.append(mask)
imgs = np.stack(imgs)
print(imgs.shape)
masks = np.stack(masks)
print(masks.shape)
labels = np.stack(labels)
print(labels.shape)

np.save('shuffled/img_segmented.npy', imgs)
np.save('shuffled/node_mask.npy', masks)
np.save('shuffled/labels.npy', labels)


