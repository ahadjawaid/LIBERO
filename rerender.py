from pointcloud import get_rgbd_image, get_point_cloud, get_colored_point_cloud
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from visualizer import Visualizer
from typing import Tuple, Union
from numpy import ndarray
from pathlib import Path
import numpy as np
import h5py
import json

def get_benchmark_demo_files(dataset_name: str, datasets_path: Path) -> list:
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[dataset_name]()
    num_tasks = benchmark_instance.get_num_tasks()
    demo_files = [datasets_path / benchmark_instance.get_task_demonstration(i) for i in range(num_tasks)]
    return demo_files

def get_demo_states_and_env_metadata(demo_file) -> Tuple[dict, ndarray]:
    with h5py.File(demo_file, "r") as f:
        env_metadata = json.loads(f["data"].attrs["env_args"])
        states = f['data']['demo_0']['states'][()]
    
    return env_metadata, states

def get_bddl_path(demo_file: Path, bddl_files_path: Union[str, Path] = None) -> Path:
    bddl_files_path = Path(bddl_files_path) if bddl_files_path else Path(get_libero_path("bddl_files"))
    bddl_path = bddl_files_path / demo_file.parent.stem / (demo_file.stem[:-5] + '.bddl')
    return bddl_path

def get_env_kwargs(env_metadata: dict, bddl_path: Union[str, Path]) -> dict:
    env_kwargs = env_metadata['env_kwargs']
    env_kwargs['controller'] = env_kwargs.pop('controller_configs')['type']
    env_kwargs['camera_depths'] = True
    env_kwargs['bddl_file_name'] = str(bddl_path)
    return env_kwargs

def rerender_observations_and_intrinsics(states, env_kwargs: dict) -> dict:
    env = OffScreenRenderEnv(**env_kwargs)
    env.reset()

    camera_height, camera_width = env_kwargs['camera_heights'], env_kwargs['camera_widths']
    camera_names = env_kwargs['camera_names']
    observations = {
        camera_name: dict(
            rgb_image=[], 
            depth_image=[], 
            intrinsic=get_camera_intrinsic_matrix(env.sim, camera_name, camera_height, camera_width),
            camera_position=get_camera_position(env.sim, camera_name)
        ) for camera_name in camera_names
    }

    for state in states:
        observation = env.set_init_state(state)
        
        for camera_name in camera_names:
            rgb_img, depth_img = observation[camera_name + "_image"], observation[camera_name + "_depth"]
            observations[camera_name]['rgb_image'].append(rgb_img)
            observations[camera_name]['depth_image'].append(depth_img)

    env.close()
    return observations

def get_camera_position(sim, camera_name: str) -> ndarray:
    camera_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.model.body_pos[camera_id]
    return camera_pos

def verticalFlip(img: ndarray) -> ndarray:
    return np.flip(img, axis=0)

def get_args() -> Tuple[str, Path]:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="libero_spatial")
    args = parser.parse_args()
    dataset_name, datasets_path = args.dataset_name, Path(args.datasets_path)

    return dataset_name, datasets_path

if __name__ == '__main__':
    dataset_name, datasets_path = get_args()
    demo_files = get_benchmark_demo_files(dataset_name, datasets_path)
    demo_file = demo_files[0]

    bddl_path = get_bddl_path(demo_file)
    env_metadata, states = get_demo_states_and_env_metadata(demo_file)
    env_kwargs = get_env_kwargs(env_metadata, bddl_path)

    observations = rerender_observations_and_intrinsics(states, env_kwargs)

    camera_names = list(observations.keys())
    camera_name = camera_names[1]
    camera_observations = observations[camera_name]
    camera_intrinsic = camera_observations['intrinsic']

    index = len(camera_observations['rgb_image']) // 2
    rgb_image = camera_observations['rgb_image'][index]
    depth_image = camera_observations['depth_image'][index]

    rgbd_image = get_rgbd_image(rgb_image, depth_image)
    camera_height, camera_width = env_kwargs['camera_heights'], env_kwargs['camera_widths']
    point_cloud = get_point_cloud(rgbd_image, camera_intrinsic, camera_height, camera_width)
    colored_points = get_colored_point_cloud(point_cloud, rgb_image)

    vis = Visualizer()
    vis.visualize_pointcloud(colored_points)