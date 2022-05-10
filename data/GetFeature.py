import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA

n_dim = 32
parent_path = 'slice/3_5/img_segmented'
data_path = 'base.csv'
img_paths = os.listdir(parent_path)

imgs = []
for img_path in img_paths:
    img_path = parent_path + '/' + img_path
    img = np.load(img_path)
    img = img[2]    # 肿瘤中心水平位切面
    img = img.reshape(-1)
    imgs.append(img)
imgs = np.stack(imgs, axis=0)

# 提取图片特征
# 先使用PCA
pca = PCA(n_components=n_dim)
features = pca.fit_transform(imgs)
np.save('pca_features.npy', features)

# 和其他特征组合
data = pd.read_csv(data_path, sep='\t')



