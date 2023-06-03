# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

# Training constants
HR_IMG_SIZE = 300
LR_IMG_SIZE = 100
SEED = 42

# Visualization constants
SAMPLE_IMAGES_VAL = [
    ['val/108070.jpg', [250, 325, 100, 175]],
    ['val/299086.jpg', [320, 395, 200, 275]],
    ['val/41069.jpg', [225, 300, 150, 225]],
    ['val/210088.jpg', [150, 225, 200, 275]],
    ['val/86000.jpg', [225, 300, 200, 275]]
]

# Dataframe constants
DF_COLUMNS = [
    'BIC_MSE',
    'GEN_MSE',
    'BIC_SSIM',
    'GEN_SSIM',
    'BIC_PSNR',
    'GEN_PSNR'
]

# Model constants
# Model only upscales to a factor of 3
# do not change unless you also change the generator
UPSCALE_FACTOR = 3
