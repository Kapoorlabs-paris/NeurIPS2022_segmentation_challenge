import os
from pathlib import Path
import concurrent
from tifffile import imread, imwrite
import numpy as np
import cv2

inputdir = Path("/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw/")
outputdir = "/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw/"
Path(outputdir).mkdir(exist_ok=True)
pattern = '*.tiff'
minsize = (512,512)

files = list(inputdir.glob(pattern))
nthreads = os.cpu_count()
def resizer(file):
    image = imread(file)
    newimage = np.zeros(minsize)
    if image.shape[0] < minsize[0] and image.shape[1] < minsize[1]:
          newimage =cv2.resize(
                image.astype('float32'), minsize)
    return newimage, file.name   
with concurrent.futures.ThreadPoolExecutor(max_workers = nthreads) as executor:
     futures = []
     for fname in files:
         futures.append(executor.submit(resizer, file = fname))
     for future in concurrent.futures.as_completed(futures):
                   newimage, name = future.result()
                   imwrite(outputdir + '/' + os.path.splitext(name)[0] + '.tiff', newimage)