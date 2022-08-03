import os
from pathlib import Path
from tifffile import imwrite, imread
import imageio
import numpy as np
from scipy import misc


inputdir = Path("/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/")
outputdir = "/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw/"
Path(outputdir).mkdir(exist_ok=True)
pattern = '*.tif'
files = list(inputdir.glob(pattern))
for fname in files:

    image = imread(fname) 
    print('in', image.shape)
    if(len(image.shape)==2):
          newimage = np.zeros([image.shape[0], image.shape[1], 3]) 
          for i in range(3): 
              newimage[:,:,i] = image
    else:
        newimage = image 
    print(newimage.shape)
    imwrite(outputdir + '/' + os.path.splitext(fname.name)[0] + '.tiff', newimage)
