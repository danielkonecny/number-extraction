import argparse
from pathlib import Path
from PIL import Image
import tensorflow as tf
from safe_gpu import safe_gpu

from crnn.dataset_factory import NumberDatasetBuilder


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


def generate_train_samples():
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


def generate_test_samples():
    directory = Path("datasets/numbers3/test")
    directory.mkdir(parents=True, exist_ok=True)

    count = 10

    dataset_builder = NumberDatasetBuilder(text_width=145, text_height=40, canvas_width=200, canvas_height=300,
                                           random_position=True, random_scale=True)
    dataset = dataset_builder(count=count, batch_size=32)

    for images, labels in dataset:
        labels = tf.sparse.to_dense(labels)
        for image, label in zip(images, labels):
            label_str = tf.strings.as_string(label)
            label_str = tf.strings.join(label_str)
            image *= 255
            image = Image.fromarray(image.numpy().astype("uint8"))
            image.save(directory / f"{label_str.numpy().decode('UTF-8')}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help="Turn on Safe GPU when run on a machine with multiple GPUs.")
    args = parser.parse_args()

    if args.gpu:
        # noinspection PyUnusedLocal
        gpu_owner = safe_gpu.GPUOwner(placeholder_fn=safe_gpu.tensorflow_placeholder)

    demonstrate()
    # generate_train_samples()
    # generate_test_samples()


if __name__ == "__main__":
    main()
