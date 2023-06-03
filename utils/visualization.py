from constants import SAMPLE_IMAGES_VAL
from data_preprocessing import downscale, upscale

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array


def plot_results(img, title, zoom, save_path='.', filename='result', save=True):
    # Plot the result with zoom-in area
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0

    # Create a new figure with a default 111 subplot.
    fig, ax = plt.subplots()
    im = ax.imshow(img_array[::-1], origin="lower")

    plt.title(title)
    # zoom-factor: 2.0, location: upper-left
    axins = zoomed_inset_axes(ax, 2, loc=2)
    axins.imshow(img_array[::-1], origin="lower")

    # Specify the limits.
    x1, x2, y1, y2 = zoom

    # Apply the x-limits.
    axins.set_xlim(x1, x2)
    # Apply the y-limits.
    axins.set_ylim(y1, y2)

    plt.yticks(visible=False)
    plt.xticks(visible=False)

    # Make the line.
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")

    if save:
        plt.savefig(f'{save_path}/{filename}.png')

    plt.show()


def process_through_model(model, img):
    # Prepare image
    img = img_to_array(img)

    # Scale from (0, 255) to (0, 1)
    img = img.astype("float32") / 255.0

    # Redimension (1, height, width, channels)
    img = np.expand_dims(img, axis=0)

    # Process through model
    output = model(img, training=False)

    # Scale back from (0, 1) to (0, 255)
    output *= 255.0

    # Redimension (height, width, channels)
    output = tf.squeeze(output)

    # Return image
    output = array_to_img(output)

    return output


def epoch_statistics(epoch, df_train_epoch, df_valid_epoch, time):
    n = epoch + 1
    print('\n\n*******************************************')
    print(f'**                EPOCH {n}                **')
    print('*******************************************')
    print(f'Time for epoch {n} is {round(time, 2)} sec')

    print('\nTraining dataset - Mean')
    print('---------------------------------------')
    print(df_train_epoch[f'EPOCH_{n}'].mean(axis=0))

    print('\nTraining dataset - Standard deviation')
    print('---------------------------------------')
    print(df_train_epoch[f'EPOCH_{n}'].std(axis=0))

    print('\nValidation dataset - Mean')
    print('---------------------------------------')
    print(df_valid_epoch[f'EPOCH_{n}'].mean(axis=0))

    print('\nValidation dataset - Standard deviation')
    print('---------------------------------------')
    print(df_valid_epoch[f'EPOCH_{n}'].std(axis=0))


def show_sample_images(epoch, model, path_to_save_results):
    i = 1
    for img in SAMPLE_IMAGES_VAL:
        hr_img = load_img(img[0])
        lr_img = downscale(hr_img)

        # High resolution image
        plot_results(
            hr_img,
            'High resolution image',
            img[1],
            path_to_save_results,
            f'Epoch_{epoch}_HR_{i}'
        )

        # Bicubic image
        bic_img = upscale(lr_img)
        plot_results(
            bic_img,
            'Upsample by Bicubic interpolation',
            img[1],
            path_to_save_results,
            f'Epoch_{epoch}_BIC_{i}'
        )

        # Generated image
        gen_img = process_through_model(model, lr_img)
        plot_results(
            gen_img,
            'Upsample by model',
            img[1],
            path_to_save_results,
            f'Epoch_{epoch}_GEN_{i}'
        )

        i += 1
