import os
import shutil
from glob import glob
from unittest import TestCase, main

import torch
from torchvision import models

from toy.hash_ckpt import _hash_ckpt


class TestHashCheckpoint(TestCase):
    def test_hash_ckpt(self):
        torch.hub.set_dir(os.getcwd())
        names = ["alexnet"]  # , "vgg11", "resnet18", "mobilenet_v2"]
        for name in names:
            shutil.rmtree(os.path.join(os.getcwd(), "checkpoints"))
            model = getattr(models, name)(pretrained=True, progress=False)
            vision_filename = glob(f"{os.getcwd()}/checkpoints/*")
            prev_hash = vision_filename[-1].split("/")[-1].strip(".py").split("-")[-1].strip(".pth")
            _, sha_hash = _hash_ckpt(vision_filename[-1])
            self.assertEqual(sha_hash[:8], prev_hash)
            model.load_state_dict(torch.load(vision_filename[-1]), strict=True)


if __name__ == "__main__":
    main()
