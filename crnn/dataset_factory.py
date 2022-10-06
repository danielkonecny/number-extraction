import os
import re

import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


class Dataset(tf.data.TextLineDataset):
    def __init__(self, filename, **kwargs):
        self.dirname = os.path.dirname(filename)
        super().__init__(filename, **kwargs)

    def parse_func(self, line):
        raise NotImplementedError

    def parse_line(self, line):
        line = tf.strings.strip(line)
        img_relative_path, label = self.parse_func(line)
        img_path = tf.strings.join([self.dirname, os.sep, img_relative_path])
        return img_path, label


class SimpleDataset(Dataset):
    def parse_func(self, line):
        split_line = tf.strings.split(line)
        img_relative_path, label = split_line[0], split_line[1]
        return img_relative_path, label


class MJSynthDataset(Dataset):
    def parse_func(self, line):
        split_line = tf.strings.split(line)
        img_relative_path = split_line[0]
        label = tf.strings.split(img_relative_path, sep='_')[1]
        return img_relative_path, label


class ICDARDataset(Dataset):
    def parse_func(self, line):
        split_line = tf.strings.split(line, sep=',')
        img_relative_path, label = split_line[0], split_line[1]
        label = tf.strings.strip(label)
        label = tf.strings.regex_replace(label, r'"', '')
        return img_relative_path, label


class DatasetBuilder:
    def __init__(self, table_path, img_shape=(32, None, 3), max_img_width=300, ignore_case=False):
        # map unknown label to 0
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.TextFileInitializer(
                table_path,
                tf.string,
                tf.lookup.TextFileIndex.WHOLE_LINE,
                tf.int64,
                tf.lookup.TextFileIndex.LINE_NUMBER
            ),
            0
        )
        self.img_shape = img_shape
        self.ignore_case = ignore_case
        if img_shape[1] is None:
            self.max_img_width = max_img_width
            self.preserve_aspect_ratio = True
        else:
            self.preserve_aspect_ratio = False

    @property
    def num_classes(self):
        return self.table.size()

    @staticmethod
    def _parse_annotation(path):
        with open(path) as f:
            line = f.readline().strip()
        if re.fullmatch(r'.*/*\d+_.+_(\d+)\.\w+ \1', line):
            return MJSynthDataset(path)
        elif re.fullmatch(r'.*/*word_\d\.\w+, ".+"', line):
            return ICDARDataset(path)
        elif re.fullmatch(r'.+\.\w+ .+', line):
            return SimpleDataset(path)
        else:
            raise ValueError('Unsupported annotation format')

    def _concatenate_ds(self, ann_paths):
        datasets = [self._parse_annotation(path) for path in ann_paths]
        concatenated_ds = datasets[0].map(datasets[0].parse_line)
        for ds in datasets[1:]:
            ds = ds.map(ds.parse_line)
            concatenated_ds = concatenated_ds.concatenate(ds)
        return concatenated_ds

    def _decode_img(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=self.img_shape[-1])
        if self.preserve_aspect_ratio:
            img_shape = tf.shape(img)
            scale_factor = self.img_shape[0] / img_shape[0]
            img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
            img_width = tf.cast(img_width, tf.int32)
        else:
            img_width = self.img_shape[1]
        img = tf.image.resize(img, (self.img_shape[0], img_width)) / 255.0
        return img, label

    # noinspection PyUnusedLocal
    def _filter_img(self, img, label):
        img_shape = tf.shape(img)
        return img_shape[1] < self.max_img_width

    def _tokenize(self, images, labels):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.table.lookup, chars)
        # TODO(hym) Waiting for official support to use RaggedTensor in keras
        tokens = tokens.to_sparse()
        return images, tokens

    def __call__(self, ann_paths, batch_size, is_training):
        ds = self._concatenate_ds(ann_paths)
        if self.ignore_case:
            ds = ds.map(lambda x, y: (x, tf.strings.lower(y)))
        if is_training:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.map(self._decode_img, tf.data.AUTOTUNE)
        if self.preserve_aspect_ratio and batch_size != 1:
            ds = ds.filter(self._filter_img)
            ds = ds.padded_batch(batch_size, drop_remainder=is_training)
        else:
            ds = ds.batch(batch_size, drop_remainder=is_training)
        ds = ds.map(self._tokenize, tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


class NumberDatasetBuilder:
    def __init__(self, text_width, text_height, canvas_width=None, canvas_height=None,
                 random_position=False, random_scale=False):
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            ),
            default_value=-1
        )
        self.text_width = text_width
        self.text_height = text_height

        if canvas_width is None:
            self.canvas_width = text_width
        else:
            self.canvas_width = canvas_width

        if canvas_height is None:
            self.canvas_height = text_height
        else:
            self.canvas_height = canvas_height

        self.random_position = random_position
        self.random_scale = random_scale

    def __call__(self, count=1, batch_size=1):
        numbers_orig = tf.random.uniform(shape=[count], minval=100_000_000, maxval=1_000_000_000, dtype=tf.dtypes.int32)
        numbers_str = tf.strings.as_string(numbers_orig)
        dataset = tf.data.Dataset.from_tensor_slices(numbers_str)

        # tf.data.Dataset uses graph execution and therefore requires only TF compatible methods.
        # Use a tf.py_function to allow for other code executed eagerly (line-by-line) in the Dataset pipeline.
        # Keep in mind that this may cause a bottleneck when loading a dataset.
        dataset = dataset.map(
            lambda numbers: (
                tf.py_function(self._image_of_number, [numbers], tf.dtypes.float32),
                numbers
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(
            lambda images, numbers: (self._data_augmentation(images), numbers),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(
            lambda images, numbers: (images, self._number_to_label(numbers)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    @property
    def num_classes(self):
        return self.table.size()

    def _image_of_number(self, number):
        image = Image.new('RGB', (self.text_width, self.text_height), color="black")
        draw = ImageDraw.Draw(image)
        # TODO - Choose random font from multiple ones.
        font = ImageFont.truetype(r'fonts/times_new_roman.ttf', 30)
        draw.text((6, 2), f"{number.numpy().decode('utf-8')}", font=font, fill="white")
        image = tf.keras.utils.img_to_array(image)
        image /= 255.
        return image

    def _number_to_label(self, number):
        number = tf.strings.unicode_split(number, 'UTF-8')
        number = tf.strings.to_number(number, tf.dtypes.int32)
        label = tf.ragged.map_flat_values(self.table.lookup, number)
        label = label.to_sparse()
        return label

    def _data_augmentation(self, image):
        if self.canvas_width != self.text_width and self.canvas_height != self.text_height:
            # Scale image of number to a smaller one.
            if self.random_scale:
                scale = tf.random.uniform(shape=[], minval=.5, maxval=1.)
                new_height = tf.cast(scale * self.text_height, dtype=tf.dtypes.int32)
                new_width = tf.cast(scale * self.text_width, dtype=tf.dtypes.int32)
            else:
                new_height = self.text_height
                new_width = self.text_width

            # Move the number to a random position on a canvas.
            if self.random_position:
                offset_height = tf.random.uniform(
                    shape=[],
                    maxval=self.canvas_height - new_height,
                    dtype=tf.dtypes.int32
                )
                offset_width = tf.random.uniform(
                    shape=[],
                    maxval=self.canvas_width - new_width,
                    dtype=tf.dtypes.int32
                )
            else:
                offset_height = 15
                offset_width = 10
            image = tf.image.resize_with_pad(image, target_height=new_height, target_width=new_width)
            image = tf.image.pad_to_bounding_box(
                image,
                offset_height=offset_height,
                offset_width=offset_width,
                target_height=self.canvas_height,
                target_width=self.canvas_width
            )

        # Add random noise to the image.
        image = tf.math.add(
            image,
            tf.random.uniform(shape=[self.canvas_height, self.canvas_width, 3], minval=.0, maxval=.5)
        )
        image = tf.math.minimum(image, 1)

        # Change contrast to a random smaller value.
        max_signed_int = 2 ** 16
        image = tf.image.stateless_random_contrast(
            image,
            .5,
            .95,
            tf.random.uniform(shape=[2], maxval=max_signed_int, dtype=tf.dtypes.int32)
        )
        return 1 - image
