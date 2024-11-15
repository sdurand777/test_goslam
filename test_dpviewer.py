
from viewerdpvo import Viewer

import torch

import time


# import argparse
# from datetime import datetime
# from pathlib import Path
#
# import gin
# import numpy as np
# import torch
# from einops import *
# from tqdm import tqdm
#
# from dataloaders.ETH3D import ETH3D
# from multi_slam.fullsystem import FullSystem
# from multi_slam.locnet import LocNet
# from multi_slam.MultiTrajectory import MultiTrajectory

class ViewerX:
    def __init__(self):
# data for viewer
        self.intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")
        self.image_ = torch.zeros(480, 640, 3, dtype=torch.uint8, device="cpu") 
        self.poses_ =  torch.zeros(100, 7, dtype=torch.float, device="cuda")
        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0
        self.points_ = torch.zeros(96*100, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(100, 96, 3, dtype=torch.uint8, device="cuda")
        # define Viewer to render SLAM
        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            self.intrinsics_)
        #time.sleep(1)

    def __call__(self):

        if self.viewer is not None:
            print("viewer update : ")
            image = torch.rand(480,640,3)
            self.viewer.update_image(image)
            # update pose
# Définir les paramètres de la distribution normale
            mean = 10.0
            std = 10.0  # Écart type pour contrôler l'amplitude du bruit
# Générer le bruit avec la même taille que self.poses_
            noise = torch.normal(mean=mean, std=std, size=self.poses_.size(), device=self.poses_.device)
# Ajouter le bruit au tensor de poses
            self.poses_ = self.poses_ + noise


viewer = None

while True:
    if viewer is None:
        viewer = ViewerX()
    print("sleep ...")
    time.sleep(1)
    viewer()





print("script test dpviewer")

print("run viewer")


time.sleep(1)

print("fin du viewer")

# while True:
#     print("en cours ...")
#     time.sleep(10)
