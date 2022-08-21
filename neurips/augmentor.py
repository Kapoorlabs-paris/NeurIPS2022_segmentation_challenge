import os
import glob
from tifffile import imread, imwrite
from vollseg import Augmentation2DC
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter


image_dir =  Path('/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw_patches_512_xl/')
label_dir = Path('/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_patch_mask_512_xl/')

aug_image_dir =  '/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw_aug/'
aug_seg_image_dir = '/gpfsscratch/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_mask_aug/'


Path(aug_image_dir).mkdir(exist_ok=True)
Path(aug_seg_image_dir).mkdir(exist_ok=True)

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
rotation_angles = [2, 4, 6, 8, 12] 
pattern = '*.tiff'
sigma = 10
mean = 0
alpha_affine = 1.2
distribution = 'Both'
filesRaw = list(image_dir.glob(pattern))
filesLabel = list(label_dir.glob(pattern))



count = 0
for fname in filesRaw:

    for secondfname in filesLabel:

        name = os.path.basename(os.path.splitext(fname)[0])
        LabelName = os.path.basename(os.path.splitext(secondfname)[0])
        if name == LabelName:
                image = imread(fname)
               
                labelimage = imread(secondfname)
                for rotate_angle in rotation_angles:
                                
                                rotate_pixels = Augmentation2DC(rotate_angle = rotate_angle)

                                aug_rotate_pixels,aug_rotate_pixels_label  = rotate_pixels.build(image = np.copy(image), labelimage = labelimage)
                                
                               
                                save_name_raw = aug_image_dir + '/' + 'rotation_' +  str(rotate_angle) + name + '.tiff'
                                save_name_seg = aug_seg_image_dir + '/' + 'rotation_' +  str(rotate_angle) + name + '.tiff'
                                if os.path.exists(save_name_raw) == False:
                                    imwrite(save_name_raw, aug_rotate_pixels.astype('float32'))
                                if os.path.exists(save_name_seg) == False:    
                                    imwrite(save_name_seg, aug_rotate_pixels_label.astype('uint16'))
                                count = count + 1   

                addnoise_pixels = Augmentation2DC(mean = mean, sigma = sigma, distribution = distribution)

                aug_addnoise_pixels,aug_addnoise_pixels_label  = addnoise_pixels.build(image = np.copy(image), labelimage = labelimage)
                
                save_name_raw = aug_image_dir + '/' + 'noise_' +  str(sigma) + name + '.tiff'
                save_name_seg = aug_seg_image_dir + '/' + 'noise_' +   str(sigma) + name + '.tiff'
                if os.path.exists(save_name_raw) == False:
                    imwrite(save_name_raw, aug_addnoise_pixels.astype('float32'))
                if os.path.exists(save_name_seg) == False:    
                    imwrite(save_name_seg, aug_addnoise_pixels_label.astype('uint16'))
                count = count + 1                
 
                adddeform_pixels = Augmentation2DC(alpha_affine = alpha_affine, sigma = sigma)

                aug_adddeform_pixels,aug_adddeform_pixels_label  = adddeform_pixels.build(image = np.copy(image), labelimage = labelimage)
                
                save_name_raw = aug_image_dir + '/' + 'deform_' +  str(sigma) + name + '.tiff'
                save_name_seg = aug_seg_image_dir + '/' + 'deform_' +   str(sigma) + name + '.tiff'
                if os.path.exists(save_name_raw) == False:
                    imwrite(save_name_raw, aug_addnoise_pixels.astype('float32'))
                if os.path.exists(save_name_seg) == False:    
                    imwrite(save_name_seg, aug_addnoise_pixels_label.astype('uint16'))
                count = count + 1  

                flip_pixels = Augmentation2DC(vertical_flip = True)

                aug_flip_pixels,aug_flip_pixels_label  = flip_pixels.build(image = np.copy(image), labelimage = labelimage)
                
                save_name_raw = aug_image_dir + '/' + 'vflip_'  + name + '.tiff'
                save_name_seg = aug_seg_image_dir + '/' + 'vflip_'  + name + '.tiff'
                if os.path.exists(save_name_raw) == False:
                    imwrite(save_name_raw, aug_addnoise_pixels.astype('float32'))
                if os.path.exists(save_name_seg) == False:    
                    imwrite(save_name_seg, aug_addnoise_pixels_label.astype('uint16'))
                count = count + 1

                flip_pixels = Augmentation2DC(horizontal_flip = True)

                aug_flip_pixels,aug_flip_pixels_label  = flip_pixels.build(image = np.copy(image), labelimage = labelimage)
                
                save_name_raw = aug_image_dir + '/' + 'hflip_'  + name + '.tiff'
                save_name_seg = aug_seg_image_dir + '/' + 'hflip_'  + name + '.tiff'
                if os.path.exists(save_name_raw) == False:
                    imwrite(save_name_raw, aug_addnoise_pixels.astype('float32'))
                if os.path.exists(save_name_seg) == False:    
                    imwrite(save_name_seg, aug_addnoise_pixels_label.astype('uint16'))
                count = count + 1