import os
import cv2
import numpy as np

from base import file_utils
from config.global_configs import BaseConfig, ProjectConfig, TFRecordBaseConfig, TFRecordConfig


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


def save_images(train_dir, shape, files):
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

    for image_f, label_f in files:
        with open(image_f, 'rb') as f:
            images = f.read()
        with open(label_f, 'rb') as f:
            labels = f.read()

        images = [d for d in images[16:]]
        images = np.array(images, dtype=np.uint8)
        images = images.reshape(shape)

        for image, (index, l) in custom_zip(images, enumerate(labels[8:])):
            label = number_dict[l]
            out_dir = os.path.join(train_dir, label)
            filename = '{}-{}.jpeg'.format(label, index)
            if ProjectConfig.getDefault().debug:
                print(out_dir, filename)
            file_utils.create_directory(out_dir)
            cv2.imwrite(os.path.join(out_dir, filename), image)


def main():
    train_image = 'train-images-idx3-ubyte'
    train_label = 'train-labels-idx1-ubyte'

    test_image = 't10k-images-idx3-ubyte'
    test_label = 't10k-labels-idx1-ubyte'

    ProjectConfig.getDefault().update(project='mnist_region_classifier', debug=True)
    TFRecordConfig.getDefault().update(TFRecordBaseConfig.UPDATE_BASE)

    download_dir = ProjectConfig.getDefault().source_image_download_dir
    extract_dir = ProjectConfig.getDefault().source_image_extract_dir
    train_dir = ProjectConfig.getDefault().source_image_train_dir
    test_dir = ProjectConfig.getDefault().source_image_test_dir
    shape = (-1, ProjectConfig.getDefault().image_width, ProjectConfig.getDefault().image_height,
             ProjectConfig.getDefault().channels)

    # step 1
    # download files
    download(download_dir, [train_image, train_label, test_image, test_label])

    # step 2
    # extract files
    download_files = [os.path.join(download_dir, file) for file in
                      [train_image, train_label, test_image, test_label]]
    extract(extract_dir, download_files)

    # step 3
    # save files

    save_images(
        train_dir,
        shape,
        [(os.path.join(extract_dir, train_image), os.path.join(extract_dir, train_label))]
    )

    save_images(
        test_dir,
        shape,
        [(os.path.join(extract_dir, test_image), os.path.join(extract_dir, test_label))]
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
