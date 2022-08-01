import os
from pathlib import Path
from tifffile import imread, imwrite
import re


inputdir = Path("/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/labels/")
outputdir = "/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/real_mask/"
Path(outputdir).mkdir(exist_ok=True)
pattern = '*.tiff'
remove_string = '_label'
files = list(inputdir.glob(pattern))
for file in files:
    search_output = re.search(remove_string, file.name)
    image = imread(file)
    if search_output is not None:
          save_name = file.name[0:search_output.start()]    
    imwrite(outputdir + '/' + save_name + '.tiff', image)