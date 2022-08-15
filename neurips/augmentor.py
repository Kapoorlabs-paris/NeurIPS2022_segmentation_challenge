import os
import glob
from tifffile import imread, imwrite
from vollseg import Augmentation2DC
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter


image_dir =  Path('/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw_patches_512_xl/')
label_dir = Path('/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_patch_mask_512_xl/')

Aug_image_dir =  '/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw_aug/'
Aug_label_dir = '/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_mask_aug/'


Path(Aug_image_dir).mkdir(exist_ok=True)
Path(Aug_label_dir).mkdir(exist_ok=True)

gauss_filter_size = 0
#choices for augmentation below are 1 or 2 or None
flip_axis= 1
shift_axis= 1
zoom_axis= 1
#shift range can be between -1 and 1 (-1 and 1 will translate the pixels completely out), zoom range > 0
shift_range= 0.2 
zoom_range= 2
rotate_axis= 1
size = (512,512)
rotate_angle= 'random'
pattern = '*.tiff'
mu = 15
filesRaw = list(image_dir.glob(pattern))
filesLabel = list(label_dir.glob(pattern))


Data = []
Label = []
count = 0
for fname in filesRaw:

    for secondfname in filesLabel:

        Name = os.path.basename(os.path.splitext(fname)[0])
        LabelName = os.path.basename(os.path.splitext(secondfname)[0])
        if Name == LabelName:
                image = imread(fname)
               
                labelimage = gaussian_filter(imread(secondfname), gauss_filter_size)
                Data.append(image)
                Label.append(labelimage)
                Data = np.asarray(Data)
                Label = np.asarray(Label)

                flip_pixels = Augmentation2DC(flip_axis = flip_axis)
                aug_flip_pixels,aug_flip_pixels_label  = flip_pixels.build(data=Data, label=Label)
                aug_flip_pixels = np.reshape(aug_flip_pixels, (512,512,3))
                aug_flip_pixels_label = np.reshape(aug_flip_pixels_label, (512,512))

                Name = 'aug_flip_pixels' + str(count)
                imwrite(Aug_image_dir + '/' + str(mu) + Name + '.tiff', aug_flip_pixels.astype('float32'))
                imwrite(Aug_label_dir + '/' + str(mu) +  Name + '.tiff', aug_flip_pixels_label.astype('uint16'))
                count = count + 1
        
                rotate_pixels = Augmentation2DC(rotate_axis = rotate_axis, rotate_angle = rotate_angle)
                aug_rotate_pixels,aug_rotate_pixels_label  = rotate_pixels.build(data=Data, label=Label)
                aug_rotate_pixels = np.reshape(aug_rotate_pixels, (512,512,3))
                aug_rotate_pixels_label = np.reshape(aug_rotate_pixels_label, (512,512))

                Name = 'aug_rotate_pixels' + str(count)
                imwrite(Aug_image_dir + '/' + str(mu) + Name + '.tiff', aug_rotate_pixels.astype('float32'))
                imwrite(Aug_label_dir + '/' + str(mu) +  Name + '.tiff', aug_rotate_pixels_label.astype('uint16'))
                count = count + 1 
                

                Data = []
                Label = []           
