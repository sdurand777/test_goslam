import torch
import os
import cv2
import droid_backends
import numpy as np
import open3d as o3d
from lietorch import SE3

import time

CAM_POINTS = np.array([
    [ 0,   0,   0],
    [-1,  -1, 1.5],
    [ 1,  -1, 1.5],
    [ 1,   1, 1.5],
    [-1,   1, 1.5],
    [-0.5, 1, 1.5],
    [ 0.5, 1, 1.5],
    [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])


def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    return result


def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES)
    )

    color = (g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
    camera_actor.paint_uniform_color(color)

    return camera_actor

def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


import threading

# visualization
def droid_visualization(video, device='cuda:0', save_root=''):
    """ DROID visualization frontend """
    torch.cuda.set_device(device)
    droid_visualization.video = video
    # liste de cameras
    droid_visualization.cameras = {}
    # liste de points
    droid_visualization.points = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0
    droid_visualization.filter_thresh = 0.5

    def increase_filter(vis):
        droid_visualization.filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def decrease_filter(vis):
        droid_visualization.filter_thresh *= 1/2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    # dirty index indices des edges du graph modifie par le BA a afficher dans le rendu
    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

        with torch.no_grad():
            with video.get_lock():
                # on recupere le dirty index de video en bloquant video pour eviter une reecriture le temps de recuperer le clone
                dirty_index, = torch.where(video.dirty.clone())

            if len(dirty_index) == 0:
                return

            video.dirty[dirty_index] = False

            images = torch.index_select(video.images, dim=0, index=dirty_index)
            images = images.cpu()[:, :, 3::8, 3::8].permute(0, 2, 3, 1)

            intrinsic = video.intrinsics[0]

            # convert poses to 4x4 matrix
            poses = torch.index_select(video.poses, dim=0, index=dirty_index)
            disps = torch.index_select(video.disps, dim=0, index=dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()

            points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsic).cpu()

            thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

            count = droid_backends.depth_filter(
                video.poses, video.disps, intrinsic, dirty_index, thresh
            )

            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 2) & (disps > 0.5 * disps.mean(dim=[1, 2], keepdim=True)))

            # parcours les edges
            for i in range(len(dirty_index)):
                # recuperation de la pose
                pose = Ps[i]
                # recuperation du node
                ix = dirty_index[i].item()

                # remove camera pour la remplacer par celle optimise
                if ix in droid_visualization.cameras:
                    vis.remove_geometry(droid_visualization.cameras[ix])
                    del droid_visualization.cameras[ix]

                # remove points associes par ceux optimise
                if ix in droid_visualization.points:
                    vis.remove_geometry(droid_visualization.points[ix])
                    del droid_visualization.points[ix]

                ### add camera actor ###
                cam_actor = create_camera_actor(True)
                cam_actor.transform(pose)

                vis.add_geometry(cam_actor)
                # stockage de la camera
                droid_visualization.cameras[ix] = cam_actor

                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()

                ### add point actor ###
                point_actor = create_point_actor(points=pts, colors=clr)
                vis.add_geometry(point_actor)
                # stockage des points
                droid_visualization.points[ix] = point_actor

            print(f'\n\n Visualization: totally {masks.sum()} valid points found among {len(dirty_index)} keyframes.\n')

            # Libération explicite de la mémoire GPU pour éviter l'accumulation
            del points, images
            torch.cuda.empty_cache()

            # hack to allow interacting with visualization during inference
            if len(droid_visualization.cameras) >= droid_visualization.warmup:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            droid_visualization.ix += 1
            vis.poll_events()
            vis.update_renderer()

            # merged_point_cloud = droid_visualization.points[0]
            # for i in range(1, len(droid_visualization.points)):
            #     merged_point_cloud += droid_visualization.points[i]
            # o3d.io.write_point_cloud("test_pcd.ply", merged_point_cloud)

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_filter)
    vis.register_key_callback(ord("A"), decrease_filter)

    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json('./src/renderoption.json')

    vis.run()
    vis.destroy_window()

def start_viewer():
    """ Initialise et lance le viewer pour la visualisation """

    print("Launch viewer")

    from viewerdpvo import Viewer  # Import du module de visualisation

    # Initialisation des intrinsics avec des valeurs par défaut
    intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

    # Instanciation du viewer avec les données de visualisation
    droid_visualization_dso.viewer = Viewer(
        droid_visualization_dso.image_,
        droid_visualization_dso.poses_,
        droid_visualization_dso.points_,
        droid_visualization_dso.colors_,
        intrinsics_
    )


    print("droid_visualization_dso.image_ : ", droid_visualization_dso.image_)

    droid_visualization_dso.viewer.update_image(droid_visualization_dso.image_ )

    #time.sleep(10)


    print("Viewer Launched")


# visualization
def droid_visualization_dso(video, device='cuda:0', save_root=''):
    """ DROID visualization frontend """

    # Configuration du périphérique CUDA
    torch.cuda.set_device(device)

    # Initialisation des attributs de l'objet `droid_visualization_dso`
    droid_visualization_dso.video = video
    droid_visualization_dso.cameras = {}  # Liste de caméras
    droid_visualization_dso.points = {}  # Liste de points
    droid_visualization_dso.warmup = 8
    droid_visualization_dso.scale = 1.0
    droid_visualization_dso.ix = 0
    droid_visualization_dso.filter_thresh = 0.5

    # Images et données pour la visualisation
    ht, wd = video.images.size(2), video.images.size(3)
    droid_visualization_dso.image_ = torch.zeros(ht, wd, 3, dtype=torch.uint8, device="cpu")
    droid_visualization_dso.poses_ = torch.zeros(1000, 7, dtype=torch.float, device="cuda")
    droid_visualization_dso.poses_[:,6] = 1.0
    droid_visualization_dso.points_ = torch.zeros(1000000, 3, dtype=torch.float, device="cuda")
    droid_visualization_dso.colors_ = torch.zeros(1000, 1000, 3, dtype=torch.uint8, device="cuda")

    # Initialisation du viewer (qui sera démarré avec `start_viewer`)
    droid_visualization_dso.viewer = None

    # Lancement du viewer
    start_viewer()

    def animation_callback():
        with torch.no_grad():
            with video.get_lock():
                # on recupere le dirty index de video en bloquant video pour eviter une reecriture le temps de recuperer le clone
                dirty_index, = torch.where(video.dirty.clone())

            if len(dirty_index) == 0:
                return

            video.dirty[dirty_index] = False

            images = torch.index_select(video.images, dim=0, index=dirty_index)
            images = images.cpu()[:, :, 3::8, 3::8].permute(0, 2, 3, 1)


            # update image
            droid_visualization_dso.viewer.update_image(images)

            intrinsic = video.intrinsics[0]

            # convert poses to 4x4 matrix
            poses = torch.index_select(video.poses, dim=0, index=dirty_index)
            disps = torch.index_select(video.disps, dim=0, index=dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()

            points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsic).cpu()

            thresh = droid_visualization_dso.filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

            count = droid_backends.depth_filter(
                video.poses, video.disps, intrinsic, dirty_index, thresh
            )

            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 2) & (disps > 0.5 * disps.mean(dim=[1, 2], keepdim=True)))
            # parcours les edges
            for i in range(len(dirty_index)):
                # recuperation de la pose
                # recuperation du node
                ix = dirty_index[i].item()
                
                # recuperation des points et des couleurs et des poses
                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
                pose = Ps[i]

            #droid_visualization.points_[:len(points)] = points[:]
            # Libération explicite de la mémoire GPU pour éviter l'accumulation
            del points, images
            torch.cuda.empty_cache()

            droid_visualization_dso.ix += 1

    animation_callback()

