import os
from pathlib import Path
from tifffile import imread, imwrite
from skimage.measure import  regionprops
import numpy as np
from scipy import ndimage
import concurrent
from skimage.morphology import binary_erosion
from skimage.morphology import remove_small_objects
class SmartPatch2D(object):


    def __init__(self, base_dir, patch_size, num_channels,  raw_dir = '/raw/',
     raw_save_dir = '/raw_patches/', erosion_iterations = 2,
     real_mask_dir = '/real_mask/',real_mask_patch_dir = '/real_patch_mask/', binary_mask_dir = '/binary_patch_mask/',
     binary_erode_mask_dir = '/binary_patch_erode_mask/',  pattern = '.tif', lower_ratio_fore_to_back = 0.3,
     upper_ratio_fore_to_back = 0.9 ):

            self.base_dir = base_dir
            self.raw_dir = raw_dir
            self.raw_save_dir = raw_save_dir
            self.patch_size = patch_size
            self.erosion_iterations = erosion_iterations 
            self.num_channels = num_channels
            self.real_mask_dir = real_mask_dir
            self.binary_mask_dir = binary_mask_dir
            self.real_mask_patch_dir = real_mask_patch_dir
            self.binary_erode_mask_dir = binary_erode_mask_dir
            self.pattern = pattern
            self.search_pattern = '*' + self.pattern
            self.lower_ratio_fore_to_back = lower_ratio_fore_to_back
            self.upper_ratio_fore_to_back = upper_ratio_fore_to_back
            
            self.create_smart_patches()


    def create_smart_patches(self):        

          
         
          Path(self.base_dir + self.raw_save_dir).mkdir(exist_ok = True)
          Path(self.base_dir + self.real_mask_patch_dir).mkdir(exist_ok = True)
          Path(self.base_dir + self.binary_mask_dir).mkdir(exist_ok = True)
          Path(self.base_dir + self.binary_erode_mask_dir).mkdir(exist_ok = True) 


          Real_Mask_path = Path(self.base_dir + self.real_mask_dir)
          RealMask = list(Real_Mask_path.glob(self.search_pattern))
          
          for fname in RealMask:
               
                            print(fname)                        
                            labelimage = imread(fname)
                            name = os.path.splitext(fname.name)[0]
                            labelimage = labelimage.astype('uint16')
                            properties = regionprops(labelimage)
                            for count, prop in enumerate(properties):
                                self.label_maker( name , fname, labelimage , count , prop )
                        
                        

    def label_maker(self, name, fname,labelimage, count, prop):
                  
                      self.valid = False
                      centroid = prop.centroid
                      x = centroid[1]
                      y = centroid[0]

                      crop_Xminus = x  - int(self.patch_size[1]/2)
                      crop_Xplus = x   + int(self.patch_size[1]/2)
                      crop_Yminus = y  - int(self.patch_size[0]/2)
                      crop_Yplus = y   + int(self.patch_size[0]/2)
                      region =(slice(int(crop_Yminus), int(crop_Yplus)),
                                                                slice(int(crop_Xminus), int(crop_Xplus)))
                      
                      self.crop_labelimage = labelimage[region] 
                      self.crop_labelimage = remove_small_objects(
                               self.crop_labelimage.astype('uint16'), min_size=10)
                      if self.crop_labelimage.shape[0] == self.patch_size[0] and self.crop_labelimage.shape[1] == self.patch_size[1]:
                            self.region_selector()

                            if self.valid:

                                imwrite(self.base_dir + self.real_mask_patch_dir + '/' + os.path.splitext(fname.name)[0] + str(count) + self.pattern, self.crop_labelimage.astype('uint16'))

                                binary_image = self.crop_labelimage > 0   
                                imwrite(self.base_dir + self.binary_mask_dir + '/' + os.path.splitext(fname.name)[0] + str(count) + self.pattern, binary_image.astype('uint16'))

                                if self.erosion_iterations > 0:
                                    eroded_crop_labelimage = erode_labels(self.crop_labelimage.astype('uint16'), self.erosion_iterations)
                                eroded_binary_image = eroded_crop_labelimage > 0   
                                imwrite(self.base_dir + self.binary_erode_mask_dir + '/' + os.path.splitext(fname.name)[0] + str(count) + self.pattern, eroded_binary_image.astype('uint16'))


                                region =(slice(int(crop_Yminus), int(crop_Yplus)),
                                                                        slice(int(crop_Xminus), int(crop_Xplus)), slice(0, self.num_channels))
                                self.raw_image = imread(Path(self.base_dir + self.raw_dir + name + self.pattern ))[region]
                                
                                imwrite(self.base_dir + self.raw_save_dir + '/' + os.path.splitext(fname.name)[0] + str(count) + self.pattern, self.raw_image)
                                
                               


    def region_selector(self):
    
        
        non_zero_indices = list(zip(*np.where(self.crop_labelimage > 0)))
 
        total_indices = list(zip(*np.where(self.crop_labelimage >= 0)))
        if len(total_indices) > 0:
           norm_foreground = len(non_zero_indices)/ len(total_indices)
           index_ratio = float(norm_foreground) 
           if index_ratio >= self.lower_ratio_fore_to_back  and index_ratio <= self.upper_ratio_fore_to_back:

               self.valid = True
         



 

def erode_label_holes(lbl_img, iterations):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (range(np.min(lbl_img), np.max(lbl_img) + 1)):
        mask = lbl_img==l
        mask_filled = binary_erosion(mask,iterations = iterations)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled    

def erode_labels(segmentation, erosion_iterations= 2):
    # create empty list where the eroded masks can be saved to
    list_of_eroded_masks = list()
    regions = regionprops(segmentation)
    erode = np.zeros(segmentation.shape)
    def erode_mask(segmentation_labels, label_id, erosion_iterations):
        
        only_current_label_id = np.where(segmentation_labels == label_id, 1, 0)
        eroded = ndimage.binary_erosion(only_current_label_id, iterations = erosion_iterations)
        relabeled_eroded = np.where(eroded == 1, label_id, 0)
        return(relabeled_eroded)

    for i in range(len(regions)):
        label_id = regions[i].label
        erode = erode + erode_mask(segmentation, label_id, erosion_iterations)

    # convert list of numpy arrays to stacked numpy array
    return erode
