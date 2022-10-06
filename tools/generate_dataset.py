import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from safe_gpu import safe_gpu


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


def demonstrate():
    dataset_builder = NumberDatasetBuilder(text_width=145, text_height=40, canvas_width=200, canvas_height=300,
                                           random_position=True, random_scale=True)

    dataset = dataset_builder(count=1, batch_size=1)

    for images, numbers in dataset:
        print("Batch:")
        print(f"- Images: {images.shape}")
        print(f"- Labels: {numbers.shape}")
        numbers = tf.sparse.to_dense(numbers)
        for image, number in zip(images, numbers):
            image *= 255
            image = Image.fromarray(image.numpy().astype("uint8"))
            image.show()
            print(f"-- Number: {number}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help="Turn on Safe GPU when run on a machine with multiple GPUs.")
    args = parser.parse_args()

    if args.gpu:
        # noinspection PyUnusedLocal
        gpu_owner = safe_gpu.GPUOwner(placeholder_fn=safe_gpu.tensorflow_placeholder)

    directory = Path("datasets/numbers3")
    directory.mkdir(parents=True, exist_ok=True)

    for i in range(100, 1_000):
        new_dir = directory / f"{i // 100}" / f"{(i % 100) // 10}" / f"{(i % 10) // 1}"
        new_dir.mkdir(parents=True, exist_ok=True)

    count = 100_000
    file_path = directory / 'annotation_train.txt'

    dataset_builder = NumberDatasetBuilder(text_width=145, text_height=40, canvas_width=200, canvas_height=300,
                                           random_position=True, random_scale=True)
    dataset = dataset_builder(count=count, batch_size=64)

    with open(file_path, 'a') as text_file:
        for images, labels in dataset:
            labels = tf.sparse.to_dense(labels)
            for image, label in zip(images, labels):
                label_str = tf.strings.as_string(label)
                label_str = tf.strings.join(label_str)
                image *= 255
                image = Image.fromarray(image.numpy().astype("uint8"))
                image.save(directory / f"{label[0]}/{label[1]}/{label[2]}/{label_str.numpy().decode('UTF-8')}.png")
                text_file.write(
                    f"{label[0]}/{label[1]}/{label[2]}/{label_str.numpy().decode('UTF-8')}.png "
                    f"{label_str.numpy().decode('UTF-8')}\n"
                )


if __name__ == "__main__":
    main()
