
from SmartPatch2D import SmartPatch2D
from config_unet import NeurIPSUnetConfig

import hydra
from hydra.core.config_store import ConfigStore


configstore = ConfigStore.instance()
configstore.store(name = 'neurips_unet_config', node = NeurIPSUnetConfig)

@hydra.main(config_path="conf", config_name='config_unet')
def main(config : NeurIPSUnetConfig):

            base_dir = config.paths.base_dir
            npz_filename = config.files.npz_filename
            raw_dir = config.paths.raw_dir
            raw_save_dir = config.paths.raw_save_dir
            real_mask_dir = config.paths.real_mask_dir 
            binary_mask_dir = config.paths.binary_mask_dir
            binary_erode_mask_dir = config.paths.binary_erode_mask_dir
            #Network training parameters
            patch_x = config.params.patch_x
            patch_y = config.params.patch_y
            pattern = config.params.pattern
            num_channels = config.params.num_channels

            SmartPatch2D(base_dir, npz_filename, (patch_y, patch_x), num_channels,  raw_dir = raw_dir, raw_save_dir = raw_save_dir,
     real_mask_dir = real_mask_dir, binary_mask_dir = binary_mask_dir, binary_erode_mask_dir = binary_erode_mask_dir,  pattern = pattern)
           


if __name__ == '__main__':
    main()             