from os import mkdir
from pathlib import Path
from tifffile import imread, imwrite
import imageio


inputdir = Path("/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/")
outputdir = Path("/gpfsstore/rech/jsy/uzj81mi/Segmentation_challenge/NeurIPS_CellSegData/Train_Labeled/raw/")
Path(outputdir).mkdir(exist_ok=True)
pattern = '*.png'
files = list(inputdir.glob(pattern))
for file in files:
    image = imageio.imread(file)
    print(image.shape)