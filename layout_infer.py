from collections.abc import Sequence

from absl import app

from tpu_graphs.baselines.layout import infer_args
from tpu_graphs.baselines.layout import infer_lib
from tensorflow.python.client import device_lib
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
print(tf.config.list_logical_devices())

def main(unused_argv) -> None:
  infer_lib.infer(infer_args.get_args())


if __name__ == '__main__':
  print("which device is using")
  print(device_lib.list_local_devices())
  app.run(main)
