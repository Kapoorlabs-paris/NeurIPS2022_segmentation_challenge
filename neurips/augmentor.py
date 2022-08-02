import os
import glob
from tifffile import imread, imwrite
from vollseg import Augmentation2DC
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter


image_dir =  Path('/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw_veto_256/')
label_dir = Path('/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_mask_veto_256/')

Aug_image_dir =  '/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/aug_raw_veto_256/'
Aug_label_dir = '/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_mask_veto_256/'

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

                rotate_pixels = Augmentation2DC(rotate_axis = rotate_axis, rotate_angle = rotate_angle)
                aug_rotate_pixels = rotate_pixels.build(data=Data, label=Label)
                aug_rotate_pixels_pair = np.asarray(next(aug_rotate_pixels))
                count = 0
                for i in range(0, aug_rotate_pixels_pair.shape[1]):
                    Name = 'rotate_pixels' + str(count)
                    imwrite(Aug_image_dir + '/' + Name + '.tiff', aug_rotate_pixels_pair[0,i,:,:].astype('float32'))
                    imwrite(Aug_label_dir + '/' + Name + '.tiff', aug_rotate_pixels_pair[1,i,:,:].astype('uint16'))
                    count = count + 1   

                flip_pixels = Augmentation2DC(flip_axis = flip_axis)
                aug_flip_pixels = flip_pixels.build(data=Data, label=Label)
                aug_flip_pixels_pair = np.asarray(next(aug_flip_pixels))
                count = 0
                for i in range(0, aug_flip_pixels_pair.shape[1]):
                    Name = 'aug_flip_pixels' + str(count)
                    imwrite(Aug_image_dir + '/' + Name + '.tiff', aug_flip_pixels_pair[0,i,:,:].astype('float32'))
                    imwrite(Aug_label_dir + '/' + Name + '.tiff', aug_flip_pixels_pair[1,i,:,:].astype('uint16'))
                    count = count + 1        

                zoom_pixels = Augmentation2DC(zoom_axis = zoom_axis, zoom_range = zoom_range)
                aug_zoom_pixels = zoom_pixels.build(data=Data, label=Label)
                aug_zoom_pixels_pair = np.asarray(next(aug_zoom_pixels))
                count = 0
                for i in range(0, aug_zoom_pixels_pair.shape[1]):
                    Name = 'aug_zoom_pixels' + str(count)
                    imwrite(Aug_image_dir + '/' + Name + '.tiff', aug_zoom_pixels_pair[0,i,:,:].astype('float32'))
                    imwrite(Aug_label_dir + '/' + Name + '.tiff', aug_zoom_pixels_pair[1,i,:,:].astype('uint16'))
                    count = count + 1   

                shift_pixels = Augmentation2DC(shift_axis = shift_axis, shift_range = shift_range)
                aug_shift_pixels = shift_pixels.build(data=Data, label=Label)
                aug_shift_pixels_pair = np.asarray(next(aug_shift_pixels))
                count = 0
                for i in range(0, aug_shift_pixels_pair.shape[1]):
                    Name = 'aug_shift_pixels' + str(count)
                    imwrite(Aug_image_dir + '/' + Name + '.tiff', aug_shift_pixels_pair[0,i,:,:].astype('float32'))
                    imwrite(Aug_label_dir + '/' + Name + '.tiff', aug_shift_pixels_pair[1,i,:,:].astype('uint16'))
                    count = count + 1    


                Data = []
                Label = []           
