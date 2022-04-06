import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.filters import roberts
from scipy import ndimage as ndi


def GetSliceMask(slice):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
   
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = slice < -200
    
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
  
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
   
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
   
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
   
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
   
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    # get_high_vals = binary == 0
    # slice[get_high_vals] = 0
    
    return binary


def GetLungMask(img):
    img = sitk.GetArrayFromImage(img)
    img = img.astype(np.float64)
    lung_mask = np.zeros(img.shape)
    n_slice = img.shape[0]  #[z, y, x]
    for i in range(n_slice):
        slice_mask = GetSliceMask(img[i])
        lung_mask[i] = slice_mask
    return lung_mask

df = pd.read_csv('labels.txt', sep='\t')
df.sort_values(by='ID', ascending=True)

# 获得所有label
labels = np.array(df['MPR'])

ids = np.array(df['ID'])
z1s = np.array(df['Z1'])
z2s = np.array(df['Z2'])

for id, z1, z2 in zip(ids, z1s, z2s):
    print(id)
    id = str(id)
    # id = '1433581'
    # 读取图片和对应mask
    img = sitk.ReadImage('nii/'+id+'.nii')
    mask = sitk.ReadImage('mask/'+id+'mask.nii')
    
    # 截取肺部slice
    img = img[:,:,z2:z1]
    mask = mask[:,:,z2:z1]
    
    # 获得lung_mask
    lung_mask = GetLungMask(img)
    
    np.save('lung_mask/'+id+'lungmask', lung_mask)
