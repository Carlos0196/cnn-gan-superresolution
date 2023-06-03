from constants import SEED, BATCH_SIZE, HR_IMG_SIZE, LR_IMG_SIZE, UPSCALE_FACTOR

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

import PIL


def resize_img(img, size):
    return tf.image.resize(img, [size, size], method="area")


def get_data_sets():
    # Get images
    train_ds = image_dataset_from_directory(
        './train',
        batch_size=BATCH_SIZE,
        image_size=(HR_IMG_SIZE, HR_IMG_SIZE),
        seed=SEED,
        label_mode=None,
    )
    valid_ds = image_dataset_from_directory(
        './val',
        batch_size=BATCH_SIZE,
        image_size=(HR_IMG_SIZE, HR_IMG_SIZE),
        seed=SEED,
        label_mode=None,
    )
    test_ds = image_dataset_from_directory(
        './test',
        batch_size=BATCH_SIZE,
        image_size=(HR_IMG_SIZE, HR_IMG_SIZE),
        seed=SEED,
        label_mode=None,
    )

    # Scale from (0, 255) to (0, 1)
    train_ds = train_ds.map(lambda x: x / 255.0)
    valid_ds = valid_ds.map(lambda x: x / 255.0)
    test_ds = test_ds.map(lambda x: x / 255.0)

    # Create (high_resolution_image, low_resolution_image) pair
    train_ds = train_ds.map(
        lambda x: (resize_img(x, LR_IMG_SIZE), x)
    )
    train_ds = train_ds.prefetch(buffer_size=32)

    valid_ds = valid_ds.map(
        lambda x: (resize_img(x, LR_IMG_SIZE), x)
    )
    valid_ds = valid_ds.prefetch(buffer_size=32)

    test_ds = test_ds.map(
        lambda x: (resize_img(x, LR_IMG_SIZE), x)
    )
    test_ds = test_ds.prefetch(buffer_size=32)

    return train_ds, valid_ds, test_ds


def downscale(img):
    h = img.size[0] // UPSCALE_FACTOR
    w = img.size[1] // UPSCALE_FACTOR
    resample = PIL.Image.BICUBIC

    return img.resize((h, w), resample)


def upscale(img):
    h = img.size[0] * UPSCALE_FACTOR
    w = img.size[1] * UPSCALE_FACTOR
    resample = PIL.Image.BICUBIC

    return img.resize((h, w), resample)
