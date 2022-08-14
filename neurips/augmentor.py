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
                noise_pixels = Augmentation2DC(mu = mu)
                aug_noise_pixels,aug_noise_pixels_label  = noise_pixels.build(data=Data, label=Label)
                
                Name = 'aug_noise_pixels' + str(count)
                imwrite(Aug_image_dir + '/' + str(mu) + Name + '.tiff', aug_noise_pixels.astype('float32'))
                imwrite(Aug_label_dir + '/' + str(mu) +  Name + '.tiff', aug_noise_pixels_label.astype('uint16'))
                count = count + 1

                

                Data = []
                Label = []           
