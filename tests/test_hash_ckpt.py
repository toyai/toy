import os
import shutil
from glob import glob
from unittest import TestCase, main

import torch
from torchvision import models

from toy.hash_ckpt import __hash_ckpt


class TestHashCheckpoint(TestCase):
    def test_hash_ckpt(self):
        torch.hub.set_dir(os.getcwd())
        names = ["alexnet", "vgg11", "resnet18", "mobilenet_v2"]
        for name in names:
            model = getattr(models, name)(pretrained=True)
            vision_filename = glob(f"{os.getcwd()}/*")
            prev_hash = vision_filename[-1].split("/").strip(".py").split("-")[-1]
            _, sha_hash = __hash_ckpt(vision_filename[-1])
            self.assertEqual(sha_hash[:8], prev_hash)
            model.load_state_dict(torch.load(vision_filename[-1]), strict=True)
            shutil.rmtree(os.path.join(os.getcwd(), "checkpoints"))


if __name__ == "__main__":
    main()
