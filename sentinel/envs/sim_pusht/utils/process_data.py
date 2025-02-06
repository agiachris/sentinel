import os
import numpy as np
import zarr
import click
from tqdm import tqdm
from attrdict import AttrDict

from sentinel.envs.sim_pusht.pusht_pc_env import PushTPCEnv


@click.command()
@click.option("--in_dir", type=str)
@click.option("--out_dir", type=str)
@click.option("--num_demos", type=int)
@click.option("--obs_mode", type=str, default="pc")
@click.option("--ac_mode", type=str, default="rel")
def main(in_dir, out_dir, num_demos, obs_mode, ac_mode):
    # load environment for replay
    args = AttrDict(
        legacy=False,
        block_cog=None,
        damping=None,
        render_size=512,
        max_episode_length=300,
        randomize_rotation=False,
        scale_low=1.0,
        scale_high=1.0,
        scale_aspect_limit=100.0,
        uniform_scaling=True,
        obs_mode=obs_mode,
        ac_mode=ac_mode,
    )
    env = PushTPCEnv(args)
    env.reset()

    # create data dir
    if os.path.exists(out_dir):
        print(f"Data saving directory already exists! Exiting.")
        exit(0)
    else:
        os.makedirs(out_dir)

    data = zarr.open(in_dir, mode="r")
    episode_ends = data["meta"]["episode_ends"][...]
    actions = data["data"]["action"][...]
    states = data["data"]["state"][...]
    images = data["data"]["img"][...]

    episode_index = 0
    episode_t = 0
    num_steps = len(states)
    if num_demos <= len(episode_ends):
        num_steps = episode_ends[num_demos]
    for i in tqdm(range(num_steps), desc="Steps"):
        if episode_ends[episode_index] == i:
            episode_index += 1
            episode_t = 0
            if episode_index == num_demos:
                break
        env._set_state(states[i])
        env.render_cache = None
        render = env.render()
        d = dict(
            pc=render["pc"],
            rgb=render["images"][0],
            action=np.concatenate(
                [
                    (
                        (actions[i] - env.agent.position)
                        if ac_mode == "rel"
                        else actions[i]
                    ),
                    [0],
                ]
            ),
            eef_pos=env._get_obs(),
        )

        save_fn = f"sim_demo_ep{episode_index:06d}_view0_t{episode_t:02d}.npz"
        np.savez(os.path.join(out_dir, save_fn), **d)

        episode_t += 1


if __name__ == "__main__":
    main()
