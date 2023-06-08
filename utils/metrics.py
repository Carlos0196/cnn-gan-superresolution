from constants import DF_COLUMNS
from data_preprocessing import upscale
from visualization import process_through_model

import pandas as pd

import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array


def calc_metrics(model, lr_img, hr_img):
    # Bicubic image
    bic_img = upscale(lr_img)

    # Generated image
    img = img_to_array(lr_img).astype("float32")

    # Redimension (1, height, width, channels)
    img = np.expand_dims(img, axis=0)

    # Process through model
    output = model(img, training=False)

    # Redimension (height, width, channels)
    output = tf.squeeze(output)

    # Return image
    gen_img = array_to_img(output)

    # Ensure equal size
    h = gen_img.size[0]
    w = gen_img.size[1]

    hr_img = hr_img.resize((h, w))
    bic_img = bic_img.resize((h, w))
    gen_img = gen_img.resize((h, w))

    hr_img = img_to_array(hr_img)
    bic_img = img_to_array(bic_img)
    gen_img = img_to_array(gen_img)

    # MSE
    mse = tf.keras.losses.MeanSquaredError()
    bic_mse = mse(hr_img, bic_img).numpy()
    gen_mse = mse(hr_img, gen_img).numpy()

    # SSIM
    bic_ssim = tf.image.ssim(hr_img, bic_img, max_val=255.0).numpy()
    gen_ssim = tf.image.ssim(hr_img, gen_img, max_val=255.0).numpy()

    # PSNR
    bic_psnr = tf.image.psnr(hr_img, bic_img, max_val=255.0).numpy()
    gen_psnr = tf.image.psnr(hr_img, gen_img, max_val=255.0).numpy()

    return [bic_mse, gen_mse, bic_ssim, gen_ssim, bic_psnr, gen_psnr]

def create_epoch_dataframe(epoch_name, columns, data):
    # Create dataframe
    items = [epoch_name]
    cols = pd.MultiIndex.from_product([items, columns])

    return pd.DataFrame(data, columns=cols)

def calc_batch_metrics(model, batch):
    batch_data = []
    for i in range(len(batch[0])):
        lr_img = array_to_img(batch[0][i])
        hr_img = array_to_img(batch[1][i])

        # Get metrics from batch
        metrics = calc_metrics(model, lr_img, hr_img)

        batch_data.append(metrics)
    return batch_data

def calc_batch_metrics_disc(generator, discriminator, batch):
    batch_data = []
    for i in range(len(batch[0])):
        lr_img = np.expand_dims(batch[0][i], axis=0)
        hr_img = np.expand_dims(batch[1][i], axis=0)

        # Generate images
        output = generator(lr_img, training=False)

        # Send images to discriminator
        real_output = discriminator(hr_img, training=False)
        fake_output = discriminator(output, training=False)

        batch_data.append([real_output, fake_output])
    return batch_data
