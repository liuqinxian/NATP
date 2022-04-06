import pandas as pd
import numpy as np
import SimpleITK as sitk


def Resample(img, mask=False):
    """
    用itk方法将原始图像resample到与目标图像一致
    :param ori_img: 原始需要对齐的itk图像
    :param target_img: 要对齐的目标itk图像
    :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
    :return:img_res_itk: 重采样好的itk图像
    使用示范：
    import SimpleITK as sitk
    target_img = sitk.ReadImage(target_img_file)
    ori_img = sitk.ReadImage(ori_img_file)
    img_r = resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear)
    """
    original_size = img.GetSize() #获取图像原始尺寸
    original_spacing = img.GetSpacing() #获取图像原始分辨率
    target_spacing = [1, 1, 1] #设置图像新的分辨率为1*1*1
    target_size = [int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
                int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
                int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))] #计算图像在新的分辨率下尺寸大小
    target_origin = img.GetOrigin()      # 目标的起点 [x,y,z]
    target_direction = img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_size)		# 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_spacing)
    # 根据需要重采样图像的情况设置不同的dype
    resamplemethod = None
    if mask == True:
        resamplemethod = sitk.sitkNearestNeighbor
        resampler.SetOutputPixelType(sitk.sitkUInt8)   # 近邻插值用于mask的，保存uint8
    else:
        resamplemethod = sitk.sitkLinear
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))    
    resampler.SetInterpolator(resamplemethod)
    resampled_img = resampler.Execute(img)  # 得到重新采样后的图像
    return resampled_img

def WindowTransform(img):
    lung_window = np.array([-1500., 500.])
    # 窗宽1000~2000 窗位-500~-700
    img = (img-lung_window[0])/(lung_window[1]-lung_window[0])
    img[img<0] = 0
    img[img>1] = 1
    # img = (img*255).astype('uint8')
    return img

df = pd.read_csv('labels.txt', sep='\t')
df.sort_values(by='ID', ascending=True)

# 获得所有label
labels = np.array(df['MPR'])

ids = np.array(df['ID'])
z1s = np.array(df['Z1'])
z2s = np.array(df['Z2'])

for id, z1, z2 in zip(ids, z1s, z2s):
    id = str(id)
    # 读取图片和对应mask
    img = sitk.ReadImage('nii/'+id+'.nii')
    mask = sitk.ReadImage('mask/'+id+'mask.nii')
    
    # 截取肺部
    img = img[:,:,z2:z1]
    mask = mask[:,:,z2:z1]
    node_mask = sitk.GetArrayFromImage(mask)
    lung_mask = np.load('lung_mask/'+id+'lungmask.npy').astype('int16')
    lung_mask[node_mask==1] = 1
    lung_mask = sitk.GetImageFromArray(lung_mask)
    lung_mask.SetSpacing(mask.GetSpacing())
    lung_mask.SetOrigin(mask.GetOrigin())
    lung_mask.SetDirection(mask.GetDirection())

    # spacing interpolation
    mask = Resample(mask, mask=True)
    mask = sitk.GetArrayFromImage(mask)
    np.save('resampled/node_mask/'+id+'nodemask.npy', mask)
    lung_mask = Resample(lung_mask, mask=True)
    lung_mask = sitk.GetArrayFromImage(lung_mask)
    np.save('resampled/lung_mask/'+id+'lungmask.npy', lung_mask)
    img = Resample(img)
    img = sitk.GetArrayFromImage(img)
    img_segmented = img
    
    np.save('resampled/img/'+id+'.npy', img)
    img = WindowTransform(img)
    np.save('normalized/img/'+id+'.npy', img)
    
    img_segmented[lung_mask==0] = -1500
    np.save('resampled/img_segmented/'+id+'.npy', img_segmented)
    img_segmented = WindowTransform(img_segmented)
    np.save('normalized/img_segmented/'+id+'.npy', img_segmented)
    

    
