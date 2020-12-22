import numpy as np
import struct


def read_labels(file_name, count):
    # open file
    with open(file_name, mode='rb') as file:
        file_content = file.read()

    # read number to test and amount of labels containted in this file
    [magic_number, labels_count] = struct.unpack('>ii', file_content[0:8])

    # if the magic_number does not match, something went wrong
    if magic_number != 0x801:
        print("Magic Number does not match. (0x{:02X})".format(magic_number))
        return

    # create format string
    # this string defines the layout of file_content
    # '>' = big endian
    # 'B' = unsigned byte
    labels_format_string = '>{}B'.format(labels_count)

    # itnerpret file_content as specified by label_format_string
    labels = np.array(struct.unpack(labels_format_string, file_content[8:]))
    print("Labels loaded.")
    return labels[0:count]


def read_images(file_name, count):
    # open file
    with open(file_name, mode='rb') as file:
        file_content = file.read()

    # load number to test against, number of images, row count and column count
    [magic_number, images_count, rows,
     cols] = struct.unpack('>iiii', file_content[0:16])

    # if magic_number does not match, something went wrong
    if magic_number != 0x0803:
        print("Magic Number does not match. (0x{:02X})".format(magic_number))
        return

    if not count is None:
        images_count = count
    # create image array
    images = np.zeros((images_count, rows, cols), np.float64)
    # define format string of one image row (image in mnist files is presented row wise)
    row_format_string = '>{}B'.format(cols)
    # define start variable of actual content
    start = 16

    for image in range(0, images_count):
        for row in range(0, rows):
            end = start + cols  # define end of image row
            # load image row
            images[image, row] = np.array(
                struct.unpack(row_format_string, file_content[start:end])) / 255

            start = end  # set start to next image

        print("Images loaded: {}".format(image + 1), end='\r')

    print()
    return images