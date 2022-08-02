import os
from pathlib import Path
import concurrent
from tifffile import imread, imwrite
import numpy as np
import cv2

inputdir = Path("/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw/")
outputdir = "/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw_resized/"
Path(outputdir).mkdir(exist_ok=True)
pattern = '*.tiff'
minsize = (256,256)

files = list(inputdir.glob(pattern))
nthreads = os.cpu_count()
def resizer(file):
    image = imread(file)
    ndims = len(image.shape)
    
    if image.shape[0] < minsize[0] or image.shape[1] < minsize[1]:
          if ndims == 3:
            shape = (max(256, image.shape[1]),max(256,image.shape[0]),3)
            newimage = np.zeros(shape)
            for i in range(0, image.shape[2]):
                newimage[:,:,i] = cv2.resize(image[:,:,i].astype('float32'), (shape[0], shape[1])) 

          else:
            shape = (max(256, image.shape[1]),max(256,image.shape[0]))  
            newimage = cv2.resize(image.astype('float32'), shape)
          
    else:
        newimage = image             
    return newimage, file.name
with concurrent.futures.ThreadPoolExecutor(max_workers = nthreads) as executor:
     futures = []
     for fname in files:
         futures.append(executor.submit(resizer, file = fname))
     for future in concurrent.futures.as_completed(futures):
                   newimage, name = future.result()
                   imwrite(outputdir + '/' + os.path.splitext(name)[0] + '.tiff', newimage)
