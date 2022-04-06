import numpy as np

mask = np.load('resampled2/mask/1169027mask.npy')
mask = mask.transpose(2, 1, 0)
print(mask.shape)
x, y, z = np.where(mask>0)
x = np.max(x) - np.min(x)
y = np.max(y) - np.min(y)
z = np.max(z) - np.min(z)
print(x,y,z)

