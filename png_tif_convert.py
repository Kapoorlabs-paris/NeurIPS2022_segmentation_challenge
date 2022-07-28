import os
from pathlib import Path
from tifffile import imwrite
import imageio


inputdir = Path("/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/")
outputdir = "/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw/"
Path(outputdir).mkdir(exist_ok=True)
pattern = '*.png'
files = list(inputdir.glob(pattern))
for file in files:
    image = imageio.imread(file)
    imwrite(outputdir + '/' + os.path.splitext(file.name)[0] + '.tiff', image)