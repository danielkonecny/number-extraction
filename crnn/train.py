import argparse
import shutil
from pathlib import Path
import yaml
from datetime import datetime

from tensorflow import keras
from safe_gpu import safe_gpu

from dataset_factory import NumberDatasetBuilder
from losses import CTCLoss
from metrics import SequenceAccuracy
from models import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, help='The config file path.')
parser.add_argument('--save_dir', type=Path, required=True, help='The path to save the models, logs, etc.')
parser.add_argument('--gpu', action='store_true', help="Turn on Safe GPU when run on a machine with multiple GPUs.")
args = parser.parse_args()

if args.gpu:
    # noinspection PyUnusedLocal
    gpu_owner = safe_gpu.GPUOwner(placeholder_fn=safe_gpu.tensorflow_placeholder)

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['train']
pprint.pprint(config)

args.save_dir.mkdir(parents=True, exist_ok=True)
if list(args.save_dir.iterdir()):
    raise ValueError(f'{args.save_dir} is not a empty folder')
shutil.copy(args.config, args.save_dir / args.config.name)

def main():
    args = parse_arguments()

    with args.config.open() as f:
        config = yaml.load(f, Loader=yaml.Loader)['train']

with strategy.scope():
    lr_schedule = keras.optimizers.schedules.CosineDecay(**config['lr_schedule'])
    model = build_model(
        dataset_builder.num_classes,
        weight=config.get('weight'),
        img_shape=config['dataset_builder']['img_shape']
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=CTCLoss(),
        metrics=[SequenceAccuracy()]
    )

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.save_dir / f'{current_time}'
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, log_dir / args.config.name)

model_prefix = '{epoch}_{val_loss:.4f}_{val_sequence_accuracy:.4f}'
model_path = f'{args.save_dir}/{model_prefix}.h5'
callbacks = [
    keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True),
    keras.callbacks.TensorBoard(log_dir=f'{args.save_dir}/logs', **config['tensorboard'])
]

model.fit(train_ds, epochs=config['epochs'], callbacks=callbacks, validation_data=val_ds)

    model = build_model(
        dataset_builder.num_classes,
        img_shape=config['dataset_builder']['img_shape']
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=CTCLoss(),
        metrics=[SequenceAccuracy()]
    )

    # model.summary()

    model_prefix = '{epoch}_{val_loss:.4f}_{val_sequence_accuracy:.4f}'
    model_path = f'{log_dir}/{model_prefix}.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True),
        keras.callbacks.TensorBoard(log_dir=f'{log_dir}/logs', **config['tensorboard'])
    ]

    model.fit(train_ds, epochs=config['epochs'], callbacks=callbacks, validation_data=val_ds)


if __name__ == "__main__":
    main()
