
from vollseg import SmartSeeds2D
from config import NeurIPSConfig
import hydra
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'neurips_config', node = NeurIPSConfig)

@hydra.main(config_path="conf", config_name='config')
def main(cfg : NeurIPSConfig):

            base_dir = cfg.paths.base_dir
            model_dir = cfg.paths.model_dir
            npz_filename = cfg.files.npz_filename
            model_name = cfg.files.model_name
            raw_dir = cfg.paths.raw_dir
            real_mask_dir = cfg.paths.real_mask_dir 
            binary_mask_dir = cfg.paths.binary_mask_dir
            binary_erode_mask_dir = cfg.paths.binary_erode_mask_dir
            #Network training parameters
            depth = cfg.params.depth
            epochs = cfg.params.epoch
            learning_rate = cfg.params.learning_rate
            batch_size = cfg.params.batch_size
            patch_x = cfg.params.patch_x
            patch_y = cfg.params.patch_y
            kern_size = cfg.params.kern_size
            n_patches_per_image = cfg.params.n_patches_per_image
            n_rays = cfg.params.n_rays
            startfilter = cfg.params.startfilter
            validation_split = cfg.params.validation_split
            n_channel_in = cfg.params.n_channel_in
            pattern = cfg.params.pattern
            use_gpu_opencl = True
            load_data_sequence = False
            generate_npz = False
            train_unet = True
            train_star = True
            train_seed_unet = True
            RGB = True

            SmartSeeds2D(base_dir = base_dir, 
                        npz_filename = npz_filename, 
                        model_name = model_name, 
                        model_dir = model_dir,
                        raw_dir = raw_dir,
                        real_mask_dir = real_mask_dir,
                        binary_mask_dir = binary_mask_dir,
                        binary_erode_mask_dir = binary_erode_mask_dir,
                        n_channel_in = n_channel_in,
                        load_data_sequence = load_data_sequence, 
                        validation_split = validation_split, 
                        n_patches_per_image = n_patches_per_image, 
                        generate_npz = generate_npz, 
                        train_unet = train_unet, 
                        train_star = train_star, 
                        train_seed_unet = train_seed_unet,
                        patch_x= patch_x, 
                        patch_y= patch_y, 
                        use_gpu = use_gpu_opencl,  
                        batch_size = batch_size, 
                        depth = depth, 
                        pattern = pattern,
                        kern_size = kern_size, 
                        startfilter = startfilter, 
                        RGB = RGB,
                        n_rays = n_rays, 
                        epochs = epochs, 
                        learning_rate = learning_rate)

if __name__ == '__main__':
    main()                        
