import os
import glob
from tifffile import imread, imwrite
from vollseg import Augmentation2DC
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter


image_dir =  Path('/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw/')
label_dir = Path('/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_mask/')

Aug_image_dir =  '/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw_resize/'
Aug_label_dir = '/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_mask_resize/'

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
size = (512,512,3)
rotate_angle= 'random'
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
               
                labelimage = gaussian_filter(imread(secondfname), gauss_filter_size)
                Data.append(image)
                Label.append(labelimage)
                Data = np.asarray(Data)
                Label = np.asarray(Label)

                resize_pixels = Augmentation2DC(size =size)
                aug_resize_pixels = resize_pixels.build(data=Data, label=Label)
                aug_resize_pixels_pair = np.asarray(next(aug_resize_pixels))
                count = 0
                for i in range(0, aug_resize_pixels_pair.shape[1]):
                    Name = 'resize_pixels' + str(count)
                    imwrite(Aug_image_dir + '/' + Name + '.tiff', aug_resize_pixels_pair[0,i,:,:].astype('float32'))
                    imwrite(Aug_label_dir + '/' + Name + '.tiff', aug_resize_pixels_pair[1,i,:,:].astype('uint16'))
                    count = count + 1   

              

                Data = []
                Label = []           
