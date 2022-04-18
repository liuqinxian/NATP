import numpy as np
import pandas as pd

total_size = 128
cube_size = 32
scale = total_size // cube_size

df = pd.read_csv('labels.txt', sep='\t')
df.sort_values(by='ID', ascending=True)

# 获得所有ID
ids = np.array(df['ID'])

for id in ids:
    id = str(id)
    mask = np.load('resampled/node_mask/'+id+'nodemask.npy')
    img = np.load('normalized/img_segmented/'+id+'.npy')
    
    # 计算肿瘤中心
    # z, y, x = np.where(mask>0)
    # z, y, x = (np.max(z)+np.min(z))//2, (np.max(y)+np.min(y))//2, (np.max(x)+np.min(x))//2
    
    # 计算肿瘤左上角坐标
    z, y, x = np.where(mask>0)
    z, y, x = np.min(z), np.min(y), np.min(x)
    
    # 将图片补齐避免超出范围
    z_size, y_size, x_size = img.shape
    pad_img = np.zeros((z_size+total_size, y_size+total_size, x_size+total_size))
    pad_img[:z_size, :y_size, :x_size] = img
    
    # 裁切八个肿瘤方块
    cubes = []
    for i in range(scale):
        for j in range(scale):
            for k in range(scale):
                cube = np.zeros((cube_size, cube_size, cube_size))
                cube = pad_img[z+k*cube_size:z+(k+1)*cube_size, y+j*cube_size:y+(j+1)*cube_size, x+i*cube_size:x+(i+1)*cube_size]
                cubes.append(cube)
    cubes = np.stack(cubes)  # [scale, cube_size, cube_size, cube_size]
    print(cubes.shape)
    np.save('cube/' + str(cube_size) + '/img_segmented/'+id+'.npy', cubes)
    
    
    # 将图片补齐避免超出范围
    z_size, y_size, x_size = img.shape
    pad_img = np.zeros((z_size+total_size, y_size+total_size, x_size+total_size))
    pad_img[:z_size, :y_size, :x_size] = mask
    
    # 裁切八个肿瘤方块
    cubes = []
    for i in range(scale):
        for j in range(scale):
            for k in range(scale):
                cube = np.zeros((cube_size, cube_size, cube_size))
                cube = pad_img[z+k*cube_size:z+(k+1)*cube_size, y+j*cube_size:y+(j+1)*cube_size, x+i*cube_size:x+(i+1)*cube_size]
                cubes.append(cube)
    cubes = np.stack(cubes)  # [scale, cube_size, cube_size, cube_size]
    print(cubes.shape)
    np.save('cube/' + str(cube_size) + '/node_mask/'+id+'nodemask.npy', cubes)