
from vollseg import VollSeg, StarDist2D, UNET
from tifffile import imread
import imageio
from pathlib import Path
import numpy as np
import os
from config_predict import NeurIPSVollSegConfig
import hydra
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'neuripsstar_config', node = NeurIPSVollSegConfig)

@hydra.main(config_path="conf", config_name='config_predict')
def main(config : NeurIPSVollSegConfig):


            image_dir = config.paths.image_dir
            pattern = config.params.pattern
            search_pattern = '*' + config.params.pattern
            unet_model_name = config.files.unet_model_name 
            star_model_name = config.files.star_model_name
            model_dir = config.paths.model_dir
            save_dir = config.paths.save_dir
            seedpool = config.params.seedpool
            min_size = config.params.min_size
            min_size_mask = config.params.min_size_mask
            max_size = config.params.max_size
            lower_perc = config.params.lower_perc
            upper_perc = config.params.upper_perc
            n_tiles = (config.params.n_tiles_y, config.params.n_tiles_x, 1)
            donormalize = config.params.donormalize
            axes = config.params.axes
            slice_merge = config.params.slice_merge
            UseProbability = config.params.UseProbability
            dounet = config.params.dounet
            RGB = config.params.RGB
            prob_thresh = config.params.prob_thresh
            nms_thresh = config.params.nms_thresh
          

            star_model = StarDist2D(config = None, name = star_model_name, basedir = model_dir)
            unet_model = UNET(config = None, name = unet_model_name, basedir = model_dir)
            Raw_path = Path(image_dir)
            Raw = list(Raw_path.glob(search_pattern))
            for fname in Raw:
                    if pattern.__contains__('tiff' or 'tif'):
                       image = imread(fname)
                    if pattern.__contains__('png'):
                       image = imageio.imread(fname) 
                    if(len(image.shape)==2):
                            newimage = np.zeros([image.shape[0], image.shape[1], 3]) 
                            for i in range(3): 
                                newimage[:,:,i] = image
                    else:
                            newimage = image
                    Name = os.path.basename(os.path.splitext(fname)[0])

                    VollSeg( newimage, 
                            unet_model = unet_model, 
                            star_model = star_model, 
                            seedpool = seedpool, 
                            axes = axes, 
                            min_size = min_size,  
                            min_size_mask = min_size_mask,
                            max_size = max_size,
                            donormalize=donormalize,
                            lower_perc= lower_perc,
                            upper_perc=upper_perc,
                            n_tiles = n_tiles, 
                            prob_thresh = prob_thresh,
                            nms_thresh = nms_thresh,
                            slice_merge = slice_merge, 
                            UseProbability = UseProbability, 
                            save_dir = save_dir, 
                            Name = Name, 
                            dounet = dounet,
                            RGB = RGB)  

if __name__ == '__main__':
    main()                        
