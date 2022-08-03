import os
from pathlib import Path
from tifffile import imread, imwrite
from skimage.measure import label, regionprops

class SmartPatch2D(object):


    def __init__(self, base_dir, npz_filename, patch_size, num_channels,  raw_dir = '/Raw/',
     raw_save_dir = '/raw_patches/', erosion_iterations = 2,
     real_mask_dir = '/real_mask/', binary_mask_dir = '/binary_patches_mask/',
     binary_erode_mask_dir = '/binary_patches_erode_mask/',  search_pattern = '.tif', lower_ratio_fore_to_back = 0.3,
     upper_ratio_fore_to_back = 0.9 ):

            self.base_dir = base_dir
            self.npz_filename = npz_filename
            self.raw_dir = raw_dir
            self.raw_save_dir = raw_save_dir
            self.patch_size = patch_size
            self.erosion_iterations = erosion_iterations 
            self.num_channels = num_channels
            self.real_mask_dir = real_mask_dir
            self.binary_mask_dir = binary_mask_dir
            self.binary_erode_mask_dir = binary_erode_mask_dir
            self.search_pattern = search_pattern
            self.lower_ratio_fore_to_back = lower_ratio_fore_to_back
            self.upper_ratio_fore_to_back = upper_ratio_fore_to_back
            


    def create_smart_patches():        

          Raw_path = Path(self.base_dir + self.raw_dir)
          Raw = list(Raw_path.glob(self.search_pattern))

          Real_Mask_path = Path(self.base_dir + self.real_mask_dir)
          RealMask = list(Real_Mask_path.glob(self.search_pattern))


          for fname in RealMask: 

                labelimage = imread(fname)
                name = os.path.splitext(fname.name)[0]
                labelimage = labelimage.astype('uint16')
                properties = regionprops(labelimage)
                for count, prop in enumerate(properties):  

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
                      self.region_selector()
                      if self.valid:
                         
                         binary_image = self.crop_labelimage > 0   
                         imwrite(self.base_dir + self.binary_mask_dir + '/' + os.path.splitext(fname.name)[0] + count + self.search_pattern, binary_image)

                         if self.erosion_iterations > 0:
                               eroded_crop_labelimage = erode_labels(self.crop_labelimage.astype('uint16'), self.erosion_iterations)
                         eroded_binary_image = eroded_crop_labelimage > 0   
                         imwrite(self.base_dir + self.binary_erode_mask_dir + '/' + os.path.splitext(fname.name)[0] + count + self.search_pattern, eroded_binary_image)


                         region =(slice(int(crop_Yminus), int(crop_Yplus)),
                                                                slice(int(crop_Xminus), int(crop_Xplus)), slice(0, self.num_channels))
                         self.raw_image = imread(Path(self.base_dir + self.raw_dir + name + self.search_pattern ))[region]
                         
                         imwrite(self.base_dir + self.raw_save_dir + '/' + os.path.splitext(fname.name)[0] + count + self.search_pattern, self.raw_image)
                         

    def region_selector(self):
    
        zero_indices = np.where(self.crop_labelimage == 0)
        self.valid = False
        non_zero_indices = np.where(self.crop_labelimage > 0)
        if (len(zero_indices) > 0):
           index_ratio = len(non_zero_indices)/len(zero_indices) 

           if index_ratio >= self.lower_ratio_fore_to_back  and index_ratio <= self.upper_ratio_fore_to_back:

               self.valid = True
         
def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask_filled = binary_fill_holes(grown_mask,**kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def dilate_label_holes(lbl_img, iterations):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (range(np.min(lbl_img), np.max(lbl_img) + 1)):
        mask = lbl_img==l
        mask_filled = binary_dilation(mask,iterations = iterations)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled    

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