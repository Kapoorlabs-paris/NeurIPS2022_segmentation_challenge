import os
from pathlib import Path
from tifffile import imwrite, imread
import numpy as np


inputdir = Path("/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw/")
outputdir = "/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw/"
Path(outputdir).mkdir(exist_ok=True)
pattern = '*.tiff'
files = list(inputdir.glob(pattern))
for file in files:
    image = imread(file)
    shape = (image.shape[0], image.shape[1])
    newimage = np.zeros(shape)
    if len(image.shape) == 2:
       newimage = image
    else:
       newimage = np.sum(image, -1)    
    imwrite(outputdir + '/' + os.path.splitext(file.name)[0] + '.tiff', newimage)