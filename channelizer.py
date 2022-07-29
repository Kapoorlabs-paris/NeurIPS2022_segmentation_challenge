import os
from pathlib import Path
import concurrent
from tifffile import imread, imwrite
import numpy as np


inputdir = Path("/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw/")
outputdir = "/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw/"
Path(outputdir).mkdir(exist_ok=True)
pattern = '*.tiff'

files = list(inputdir.glob(pattern))
nthreads = os.cpu_count()
def channelizer(file):
    image = imread(file)
    shape = (image.shape[0], image.shape[1])
    newimage = np.zeros(shape)
    if len(image.shape) == 2:
       newimage = image
    else:
       newimage = np.sum(image, -1)
    return newimage, file.name   
with concurrent.futures.ThreadPoolExecutor(max_workers = nthreads) as executor:
     futures = []
     for fname in files:
         futures.append(executor.submit(channelizer, fname = fname))
     for future in concurrent.futures.as_completed(futures):
                   newimage, name = future.result()
                   imwrite(outputdir + '/' + os.path.splitext(name)[0] + '.tiff', newimage)
