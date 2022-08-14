
from vollseg import SmartSeeds2D
from config_star import NeurIPSStarConfig
import hydra
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'neuripsstar_config', node = NeurIPSStarConfig)

@hydra.main(config_path="conf", config_name='config_star')
def main(config : NeurIPSStarConfig):

            base_dir = config.paths.base_dir
            model_dir = config.paths.model_dir
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
            RGB = config.params.RGB
            n_rays = config.params.n_rays
            def_shape_x = config.params.def_shape_x 
            def_shape_y = config.params.def_shape_y
            def_shape_c = config.params.def_shape_c
            def_shape = (def_shape_x, def_shape_y, def_shape_c)
            def_label_shape = (def_shape_x, def_shape_y)
            SmartSeeds2D(base_dir = base_dir, 
                        model_name = model_name, 
                        model_dir = model_dir,
                        raw_dir = raw_dir,
                        real_mask_dir = real_mask_dir,
                        binary_mask_dir = binary_mask_dir,
                        binary_erode_mask_dir = binary_erode_mask_dir,
                        n_channel_in = n_channel_in,
                        load_data_sequence = False, 
                        validation_split = validation_split, 
                        patch_x= patch_x, 
                        patch_y= patch_y, 
                        use_gpu = False,  
                        batch_size = batch_size, 
                        depth = depth, 
                        def_shape = def_shape,
                        def_label_shape = def_label_shape,
                        pattern = pattern,
                        kern_size = kern_size, 
                        startfilter = startfilter, 
                        RGB = RGB,
                        train_star = True,
                        train_unet = False,
                        train_seed_unet = False,
                        n_rays = n_rays, 
                        epochs = epochs, 
                        learning_rate = learning_rate)

if __name__ == '__main__':
    main()                        
