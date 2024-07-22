from rerender import *
from tqdm.auto import tqdm
import h5py

def main(datasets_path: str, dataset_name: str):
    demo_files = get_benchmark_demo_files(datasets_path, dataset_name)

    for demo_file in tqdm(demo_files):
        with h5py.File(demo_file, 'r+') as f:
            data_grp = f['data']
            demo_names = list(data_grp.keys())

            for demo_name in tqdm(demo_names):
                demo_grp = data_grp[demo_name]
                obs_grp = demo_grp['obs']

                states, env_metadata = get_demo_states_and_env_metadata(demo_file, demo_name)
                bddl_path = get_bddl_path(demo_file)
                env_kwargs = get_env_kwargs(env_metadata, bddl_path)
                observations = get_rerendered_observations_and_intrinsics(states, env_kwargs)

                for key in obs_grp.keys():
                    del obs_grp[key]

                for key in observations:
                    obs_grp.create_dataset(key, data=observations[key])

if __name__ == '__main__':
    datasets_path, dataset_name = get_args()
    main(datasets_path, dataset_name)