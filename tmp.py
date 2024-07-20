from libero.libero import benchmark, get_libero_path, set_libero_default_path
import h5py
from libero.libero.utils.dataset_utils import get_dataset_info
from IPython.display import HTML
import imageio
from pathlib import Path
import json
from libero.libero.envs import OffScreenRenderEnv
from IPython.display import display
from PIL import Image
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
import torch
import torchvision
import libero.libero.utils.download_utils as download_utils

from flask import Flask, render_template_string
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.cm as cm
from termcolor import cprint
import os

import matplotlib.pyplot as plt
import open3d as o3d
    
class Visualizer:
    def __init__(self):
        self.app = Flask(__name__)
        self.pointclouds = []
        
    def _generate_trace(self, pointcloud, color:tuple=None, size=5, opacity=0.7):
        x_coords = pointcloud[:, 0]
        y_coords = pointcloud[:, 1]
        z_coords = pointcloud[:, 2]

        if pointcloud.shape[1] == 3:
            if color is None:
                # design a colorful point cloud based on 3d coordinates
                # Normalize coordinates to range [0, 1]
                min_coords = pointcloud.min(axis=0)
                max_coords = pointcloud.max(axis=0)
                normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)
                try:
                    # Use normalized coordinates as RGB values
                    colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in normalized_coords]
                except: # maybe meet NaN error
                    # use simple cyan color
                    colors = ['rgb(0,255,255)' for _ in range(len(x_coords))]
            else:    
                colors = ['rgb({},{},{})'.format(color[0], color[1], color[2]) for _ in range(len(x_coords))]
        else:
            colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in pointcloud[:, 3:6]]

        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=size,
                opacity=opacity,
                color=colors
            )
        )


    def colorize(self, pointcloud):
        if pointcloud.shape[1] == 3:

            # design a colorful point cloud based on 3d coordinates
            # Normalize coordinates to range [0, 1]
            min_coords = pointcloud.min(axis=0)
            max_coords = pointcloud.max(axis=0)
            normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)
            try:
                # Use normalized coordinates as RGB values
                colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in normalized_coords]
            except: # maybe meet NaN error
                # use simple cyan color
                x_coords = pointcloud[:, 0]
                colors = ['rgb(0,255,255)' for _ in range(len(x_coords))]

        else:
            colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in pointcloud[:, 3:6]]
        return colors
    

    def visualize_pointcloud(self, pointcloud, color:tuple=None):
        trace = self._generate_trace(pointcloud, color=color, size=6, opacity=1.0)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout)
        
        fig.update_layout(
            
            scene=dict(
                # aspectmode='cube', 
                xaxis=dict(
                    showbackground=False,  # 隐藏背景网格
                    showgrid=True,        # 隐藏网格
                    showline=True,         # 显示轴线
                    linecolor='grey',      # 设置轴线颜色为灰色
                    zerolinecolor='grey',  # 设置0线颜色为灰色
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                    
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                bgcolor='white'  # 设置背景色为白色
            )
        )
        div = pio.to_html(fig, full_html=False)

        @self.app.route('/')
        def index():
            return render_template_string('''<div>{{ div|safe }}</div>''', div=div)
        
        self.app.run(debug=True, use_reloader=False)

    def visualize_pointcloud_and_save(self, pointcloud, color:tuple=None, save_path=None):
        trace = self._generate_trace(pointcloud, color=color, size=6, opacity=1.0)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout)
        
        fig.update_layout(
            
            scene=dict(
                # aspectmode='cube', 
                xaxis=dict(
                    showbackground=False,  # 隐藏背景网格
                    showgrid=True,        # 隐藏网格
                    showline=True,         # 显示轴线
                    linecolor='grey',      # 设置轴线颜色为灰色
                    zerolinecolor='grey',  # 设置0线颜色为灰色
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                    
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                bgcolor='white'  # 设置背景色为白色
            )
        )
        # save
        fig.write_image(save_path, width=800, height=600)
        

    def save_visualization_to_file(self, pointcloud, file_path, color:tuple=None):
        # visualize pointcloud and save as html
        trace = self._generate_trace(pointcloud, color=color)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig_html = pio.to_html(go.Figure(data=[trace], layout=layout), full_html=True)

        with open(file_path, 'w') as file:
            file.write(fig_html)
        print(f"Visualization saved to {file_path}")

def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def verticalFlip(img):
    return np.flip(img, axis=0)

def get_camera_position(sim, camera_name):
    camera_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.model.body_pos[camera_id]
    return camera_pos

def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    # This function is lifted directly from scipy source code
    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat

def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

bddl_files_default_path = get_libero_path("bddl_files")
datasets_default_path = get_libero_path("datasets")

benchmark_dict = benchmark.get_benchmark_dict()

datasets = "libero_spatial" # Can specify "all", "libero_goal", "libero_spatial", "libero_object", "libero_100"
benchmark_instance = benchmark_dict[datasets]()
num_tasks = benchmark_instance.get_num_tasks()
demo_files = [os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(i)) for i in range(num_tasks)]

init_states = benchmark_instance.get_task_init_states(0)

demo_file = Path(demo_files[0])
bddl_path = str(Path(bddl_files_default_path) / demo_file.parent.stem / (demo_file.stem[:-5] + '.bddl'))

f = h5py.File(demo_file, "r")

env_metadata = json.loads(f["data"].attrs["env_args"])
env_kwargs = env_metadata['env_kwargs']
env_kwargs['controller'] = env_kwargs.pop('controller_configs')['type']
env_kwargs['camera_depths'] = True
env_kwargs['bddl_file_name'] = bddl_path

camera_height, camera_width = env_kwargs['camera_heights'], env_kwargs['camera_widths']

env = OffScreenRenderEnv(**env_kwargs)
env.seed(0)



data = f['data']
demo = data['demo_0']
states = demo['states'][()]
f.close()

for state in states:
    obs = env.set_init_state(init_states[0])
    obs, _, _, _ = env.step([0.] * 7)
    break

camera_names = [name.replace("_depth", "") for name in obs.keys() if 'depth' in name]
    
o3d_clouds = []
rgb_imgs = []
camera_name = camera_names[1]

rgb_img, depth_img = obs[camera_name + "_image"], obs[camera_name + "_depth"]
intrinsic_matrix = get_camera_intrinsic_matrix(env.sim, camera_name, camera_height, camera_width)
cam_pos = get_camera_position(env.sim, camera_name)
od_cammat = cammat2o3d(intrinsic_matrix, camera_height, camera_width)
od_depth = o3d.geometry.Image(depth_img)
o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
c2b_r = rotMatList2NPRotMat(intrinsic_matrix)
camera_id = env.sim.model.camera_name2id(camera_name)
c2b_r = rotMatList2NPRotMat(env.sim.model.cam_mat0[camera_id])
b2w_r = quat2Mat([0, 1, 0, 0])
c2w_r = np.matmul(c2b_r, b2w_r)
c2w = posRotMat2Mat(cam_pos, c2w_r)
transformed_cloud = o3d_cloud.transform(c2w)
o3d_clouds.append(transformed_cloud)
rgb_imgs.append(rgb_img)


combined_cloud = o3d.geometry.PointCloud()
cloud = o3d_clouds[0]
combined_cloud += cloud
combined_cloud_points = np.asarray(combined_cloud.points)
combined_cloud_colors = np.concatenate(rgb_imgs).reshape(-1, 3)
combined_cloud = np.concatenate((combined_cloud_points, combined_cloud_colors), axis=1)

vis = Visualizer()
vis.visualize_pointcloud(combined_cloud)