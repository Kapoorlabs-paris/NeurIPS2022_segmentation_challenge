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
outputdirs = ["/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw_resized_256/",
"/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_mask_resized_256/",
"/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/binary_mask_resized_256/",
"/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/binary_erode_mask_resized_256/"
]

assert len(inputdirs) == len(outputdirs)
pattern = '*.tiff'
minsize = (256,256)
nthreads = os.cpu_count()
for i in range(len(inputdirs)):

                inputdir = inputdirs[i]
                outputdir = outputdirs[i]

                Path(outputdir).mkdir(exist_ok=True)
                files = list(inputdir.glob(pattern))
                def resizer(file):
                    image = imread(file)
                    ndims = len(image.shape)
                    
                    if image.shape[0] < minsize[0] or image.shape[1] < minsize[1]:
                        if ndims == 3:
                            shape = (max(256, image.shape[0]),max(256,image.shape[1]),3)
                            newimage = np.zeros(shape)
                            for i in range(0, image.shape[2]):
                                newimage[:,:,i] = cv2.resize(image[:,:,i].astype('float32'), (shape[1], shape[0])) 

                        else:
                            shape = (max(256, image.shape[1]),max(256,image.shape[0]))  
                            newimage = cv2.resize(image.astype('float32'), (shape[1], shape[0]))
                        
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
