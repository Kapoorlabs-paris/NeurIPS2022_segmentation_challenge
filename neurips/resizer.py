import os
import glob
from tifffile import imread, imwrite
from vollseg import Augmentation2DC, image_pixel_duplicator
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import label

image_dir =  Path('/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw/')
label_dir = Path('/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_mask/')

Aug_image_dir =  '/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw_resize_xl/'
Aug_label_dir = '/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_mask_resize_xl/'

Path(Aug_image_dir).mkdir(exist_ok=True)
Path(Aug_label_dir).mkdir(exist_ok=True)


size_raw = (512,512,3)
size_label = (512,512)

pattern = '*.tiff'

filesRaw = list(image_dir.glob(pattern))
filesLabel = list(label_dir.glob(pattern))


Data = []
Label = []
for fname in filesRaw:

    for secondfname in filesLabel:

        Name = os.path.basename(os.path.splitext(fname)[0])
        LabelName = os.path.basename(os.path.splitext(secondfname)[0])
        if Name == LabelName:
                image = imread(fname)
                labelimage = imread(secondfname)
                if image.shape[0] < size_raw[0] or image.shape[1] < size_raw[1]:
                        image = image_pixel_duplicator(image, size_raw)
                        labelimage = label(image_pixel_duplicator(labelimage, size_label)) 
                imwrite(Aug_image_dir + '/' + Name + '.tiff', image.astype('float32'))
                imwrite(Aug_label_dir + '/' + Name + '.tiff', labelimage.astype('uint16'))
                           
