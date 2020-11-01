"""
Serialization tutorial: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
Serialization example of Pascal VOC dataset: https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py
"""

import tensorflow as tf
from object_detection.utils import dataset_util
import glob
from os import path
import xml.etree.ElementTree as ET
import random
import logging


class Example:
    class_name_dict = {}
    class_count = 0

    def __init__(self, filename, width, height):
        self.filename = filename
        self.height = height
        self.width = width

        self.xmins = []
        self.xmaxs = []
        self.ymins = []
        self.ymaxs = []

        self.class_names = []
        self.class_ids = []

    def add_xmin(self, xmin):
        xmin_normalized = xmin / self.width
        self.xmins.append(xmin_normalized)

    def add_xmax(self, xmax):
        xmax_normalized = xmax / self.width
        self.xmaxs.append(xmax_normalized)

    def add_ymin(self, ymin):
        ymin_normalized = ymin / self.height
        self.ymins.append(ymin_normalized)

    def add_ymax(self, ymax):
        ymax_normalized = ymax / self.height
        self.ymaxs.append(ymax_normalized)

    def add_class_name(self, class_name):
        if class_name not in Example.class_name_dict:
            Example.class_count += 1
            Example.class_name_dict[class_name] = Example.class_count

        self.class_names.append(class_name.encode('utf8'))
        self.class_ids.append(Example.class_name_dict[class_name])

    @staticmethod
    def create_label_map(output_labelmap_path):
        keys = Example.class_name_dict.keys()
        label_map_string = ""
        for key in keys:
            id = Example.class_name_dict[key]
            name = key
            item_string = \
                "item { \n" \
                f"  id: {id}\n" \
                f"  name: \"{name}\"\n" \
                "}\n\n"
            label_map_string += item_string

        with open(output_labelmap_path, "w") as text_file:
            print(label_map_string, file=text_file)


def load_data(annotation_folder_path):
    xml_file_paths = glob.glob(path.join(annotation_folder_path, '*.xml'))
    examples = []
    for xml_file_path in xml_file_paths:
        logging.info(f'Loading data from {xml_file_path}')
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        objects = root.findall('object')
        example = Example(
            filename=root.find('filename').text,
            width=int(root.find('size')[0].text),
            height=int(root.find('size')[1].text))

        for object in objects:
            # ToDo: Change indexes to strings
            example.add_xmin(int(object[5][0].text))
            example.add_ymin(int(object[5][1].text))
            example.add_xmax(int(object[5][2].text))
            example.add_ymax(int(object[5][3].text))

            example.add_class_name(object[0].text)

        examples.append(example)

    return examples


def create_tf_example(example, images_folder_path):
    logging.info(f'Creating TFRecord of file:  {example.filename}')
    height = example.height  # Image height
    width = example.width  # Image width
    filename = example.filename.encode('utf8')  # Filename of the image. Empty if image is not from file
    encoded_image_data = encode_image(full_img_path=path.join(images_folder_path, example.filename))

    image_format = b'jpeg'  # b'jpeg' or b'png'

    xmins = example.xmins  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = example.xmaxs  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = example.ymins  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = example.ymaxs  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = example.class_names  # List of string class name of bounding box (1 per box)
    classes = example.class_ids  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def encode_image(full_img_path):

    full_img_path = full_img_path.replace("png", "jpg")
    with tf.gfile.GFile(full_img_path, 'rb') as fid:
        encoded_image_data = fid.read()

    return encoded_image_data


def split_dataset(examples, train_ratio=0.8):
    random.shuffle(examples)
    split_index = int(len(examples) * train_ratio)

    train_examples = examples[:split_index]
    validation_examples = examples[split_index:]

    return train_examples, validation_examples


def create_tf_record_file(examples, images_folder_path, output_file_path):
    writer = tf.python_io.TFRecordWriter(output_file_path)
    for example in examples:
        tf_example = create_tf_example(example, images_folder_path)
        writer.write(tf_example.SerializeToString())
    writer.close()


flags = tf.app.flags
flags.DEFINE_string('annotation_folder_path', '', 'Path to the annotation folder')
flags.DEFINE_string('images_folder_path', '', 'Path to the image folder')
flags.DEFINE_string('output_folder_path', '', 'Folder where will be generated ')
FLAGS = flags.FLAGS


def main(_):
    annotation_folder_path = r"C:\ObjectDetection\dataset\archive\annotations"
    images_folder_path = r"C:\ObjectDetection\dataset\archive\imagesJPG"
    output_folder_path = r"C:\ObjectDetection\dataset\archive\facemasksJpg"
    # annotation_folder_path = FLAGS.annotation_folder_path
    # images_folder_path = FLAGS.images_folder_path
    # output_folder_path = FLAGS.output_folder_path

    train_ratio = 0.8

    examples = load_data(annotation_folder_path)
    (train_examples, validation_examples) = split_dataset(examples=examples, train_ratio=train_ratio)

    create_tf_record_file(train_examples, images_folder_path, path.join(output_folder_path, "train.record"))
    create_tf_record_file(validation_examples, images_folder_path, path.join(output_folder_path, "eval.record"))
    Example.create_label_map(output_labelmap_path=path.join(output_folder_path, "label_map.txt"))


if __name__ == '__main__':
    tf.app.run()
