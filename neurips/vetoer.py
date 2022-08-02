import os
from pathlib import Path
import concurrent
from tifffile import imread, imwrite
import numpy as np
import cv2

inputdirs = [Path("/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw/"),
Path("/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_mask/"),
Path("/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/binary_mask/"),
Path("/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/binary_erode_mask/")
]
outputdirs = ["/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw_veto_1024/",
"/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_mask_veto_1024/",
"/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/binary_mask_veto_1024/",
"/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/binary_erode_mask_veto_1024/"
]

assert len(inputdirs) == len(outputdirs)
pattern = '*.tiff'
minsize = (1024,1024)
nthreads = os.cpu_count()
for i in range(len(inputdirs)):

                outputdir = outputdirs[i]
                inputdir = inputdirs[i]
                Path(outputdir).mkdir(exist_ok=True)
                files = list(inputdir.glob(pattern))
                def vetoer(file):
                    image = imread(file)
                    if image.shape[0] >= minsize[0] or image.shape[1] >= minsize[1]:
                        newimage = image             
                        return newimage, file.name
                    else:
                        return None, None    

                with concurrent.futures.ThreadPoolExecutor(max_workers = nthreads) as executor:
                    futures = []
                    for fname in files:
                        futures.append(executor.submit(vetoer, file = fname))
                    for future in concurrent.futures.as_completed(futures):
                                newimage, name = future.result()
                                if newimage is not None:
                                      imwrite(outputdir + '/' + os.path.splitext(name)[0] + '.tiff', newimage)
