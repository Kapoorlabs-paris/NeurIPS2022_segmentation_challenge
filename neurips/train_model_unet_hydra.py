
from vollseg import SmartSeeds2D
from config_unet import NeurIPSUnetConfig

import hydra
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'neurips_unet_config', node = NeurIPSUnetConfig)

@hydra.main(config_path="conf", config_name='config_unet')
def main(config : NeurIPSUnetConfig):

            base_dir = config.paths.base_dir
            model_dir = config.paths.model_dir
            npz_filename = config.files.npz_filename
            model_name = config.files.model_name
            raw_dir = config.paths.raw_dir
            real_mask_dir = config.paths.real_mask_dir
            binary_mask_dir = config.paths.binary_mask_dir
            binary_erode_mask_dir = config.paths.binary_erode_mask_dir
            #Network training parameters
            depth = config.params.depth
            epochs = config.params.epoch
            learning_rate = config.params.learning_rate
            batch_size = config.params.batch_size
            patch_x = config.params.patch_x
            patch_y = config.params.patch_y
            kern_size = config.params.kern_size
            startfilter = config.params.startfilter
            validation_split = config.params.validation_split
            n_channel_in = config.params.n_channel_in
            pattern = config.params.pattern
            generate_npz = config.params.generate_npz
            RGB = config.params.RGB
            train_unet = config.params.train_unet
            train_seed_unet = config.params.train_seed_unet
            n_patches_per_image = config.params.n_patches_per_image
            

            SmartSeeds2D(base_dir = base_dir, 
                        npz_filename = npz_filename, 
                        model_name = model_name, 
                        model_dir = model_dir,
                        raw_dir = raw_dir,
                        real_mask_dir = real_mask_dir,
                        binary_mask_dir = binary_mask_dir,
                        binary_erode_mask_dir = binary_erode_mask_dir,
                        n_channel_in = n_channel_in,
                        validation_split = validation_split, 
                        n_patches_per_image = n_patches_per_image,
                        generate_npz = generate_npz, 
                        train_unet = train_unet, 
                        train_seed_unet = train_seed_unet,
                        train_star = False,
                        patch_x= patch_x, 
                        patch_y= patch_y, 
                        batch_size = batch_size, 
                        depth = depth, 
                        pattern = pattern,
                        kern_size = kern_size, 
                        startfilter = startfilter, 
                        RGB = RGB,
                        epochs = epochs, 
                        learning_rate = learning_rate)

if __name__ == '__main__':
    main()                        
