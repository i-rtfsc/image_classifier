import os
import cv2
import numpy as np

from base import file_utils
from config.global_configs import MnistConfig


def download(target_dir, files):
    for f in files:
        cmd = 'wget --no-check-certificate http://yann.lecun.com/exdb/mnist/{}.gz -P {}'.format(f, target_dir)
        os.system(cmd)


def extract(target_dir, files):
    file_utils.create_directory(target_dir)
    for f in files:
        cmd = 'gunzip -c {}.gz > {}/{}'.format(f, target_dir, os.path.basename(f))
        os.system(cmd)


def custom_zip(seq1, seq2):
    it1 = iter(seq1)
    it2 = iter(seq2)
    while True:
        try:
            yield next(it1), next(it2)
        except StopIteration:
            return


def save_images(number_dict, train_image, train_label, test_image, test_label):
    for image_f, label_f in [(train_image, train_label), (test_image, test_label)]:
        with open(image_f, 'rb') as f:
            images = f.read()
        with open(label_f, 'rb') as f:
            labels = f.read()

        images = [d for d in images[16:]]
        images = np.array(images, dtype=np.uint8)
        images = images.reshape((-1, MnistConfig.IMAGE_WIDTH, MnistConfig.IMAGE_HEIGHT, MnistConfig.CHANNELS))

        for (_, image), (k, l) in custom_zip(enumerate(images), enumerate(labels[8:])):
            label = number_dict[l]
            outdir = os.path.join(MnistConfig.MNIST_IMAGE_TRAIN, label)
            filename = '{}-{}.png'.format(label, k)
            print(outdir, filename)
            file_utils.create_directory(outdir)

            cv2.imwrite(os.path.join(outdir, filename), image)


def save_images2(number_dict, target_dir, train_image, train_label):
    for image_f, label_f in [(train_image, train_label)]:
        with open(image_f, 'rb') as f:
            images = f.read()
        with open(label_f, 'rb') as f:
            labels = f.read()

        images = [d for d in images[16:]]
        images = np.array(images, dtype=np.uint8)
        images = images.reshape((-1, 28, 28))

        for (k, image), (_, l) in custom_zip(enumerate(images), enumerate(labels[8:])):
            label = number_dict[l]
            outdir = os.path.join(target_dir, label)
            filename = '{}-{}.png'.format(label, k)
            print(outdir, filename)
            file_utils.create_directory(outdir)

            cv2.imwrite(os.path.join(outdir, filename), image)


def main():
    train_image = 'train-images-idx3-ubyte'
    train_label = 'train-labels-idx1-ubyte'

    test_image = 't10k-images-idx3-ubyte'
    test_label = 't10k-labels-idx1-ubyte'

    # download(MnistConfig.MNIST_IMAGE_DOWNLOAD, [train_image, train_label, test_image, test_label])
    #
    # download_files = [os.path.join(MnistConfig.MNIST_IMAGE_DOWNLOAD, file) for file in
    #                   [train_image, train_label, test_image, test_label]]
    # extract(MnistConfig.MNIST_IMAGE_EXTRACT, download_files)

    number_dict = dict()
    number_dict[0] = 'zero'
    number_dict[1] = 'one'
    number_dict[2] = 'two'
    number_dict[3] = 'three'
    number_dict[4] = 'four'
    number_dict[5] = 'five'
    number_dict[6] = 'six'
    number_dict[7] = 'seven'
    number_dict[8] = 'eight'
    number_dict[9] = 'nine'

    save_images(number_dict,
                os.path.join(MnistConfig.MNIST_IMAGE_EXTRACT, train_image),
                os.path.join(MnistConfig.MNIST_IMAGE_EXTRACT, train_label),
                os.path.join(MnistConfig.MNIST_IMAGE_EXTRACT, test_image),
                os.path.join(MnistConfig.MNIST_IMAGE_EXTRACT, test_label)
                )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
