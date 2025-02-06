from gym import spaces
import numpy as np

from sentinel.envs.sim_pusht.pusht_env import PushTEnv


class PushTPCEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, args, rng=None, rng_act=None):
        self.args = args

        super().__init__(
            seed=args.seed,
            legacy=args.legacy,
            block_cog=args.block_cog,
            damping=args.damping,
            render_size=args.render_size,
            max_episode_length=args.max_episode_length,
            randomize_rotation=args.randomize_rotation,
            scale_low=args.scale_low,
            scale_high=args.scale_high,
            scale_aspect_limit=args.scale_aspect_limit,
            uniform_scaling=args.uniform_scaling,
            randomize_position=(
                args.randomize_position
                if hasattr(args, "randomize_position")
                else False
            ),
            rand_pos_scale=(
                args.rand_pos_scale if hasattr(args, "rand_pos_scale") else 0.0
            ),
            render_action=getattr(args, "render_action", True),
            term_on_success=getattr(args, "term_on_success", True),
        )

        # agent position, goal position
        ws = self.window_size
        self.obs_mode = args.obs_mode
        if self.obs_mode == "pc":
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float64),
                high=np.array([ws, ws, ws, ws, ws, ws], dtype=np.float64),
                shape=(6,),
                dtype=np.float64,
            )
        elif self.obs_mode == "pc2":
            # goal represented by two points
            # designed for representations that are equivariant to positions
            self.observation_space = spaces.Box(
                low=np.array([0] * 9, dtype=np.float64),
                high=np.array([ws] * 9, dtype=np.float64),
                shape=(9,),
                dtype=np.float64,
            )

        self.ac_mode = args.ac_mode
        if self.ac_mode == "rel":
            self.action_space = spaces.Box(
                low=np.array([-ws / 2, -ws / 2, -ws / 2], dtype=np.float64),
                high=np.array([ws / 2, ws / 2, ws / 2], dtype=np.float64),
                shape=(3,),
                dtype=np.float64,
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([0, 0, 0], dtype=np.float64),
                high=np.array([ws, ws, ws], dtype=np.float64),
                shape=(3,),
                dtype=np.float64,
            )

        self.render_cache = None

    def _get_obs(self):
        # returns observation with shape (1, 6)
        if self.obs_mode == "state":
            return super()._get_obs()
        else:
            agent_pos = np.array(tuple(self.agent.position) + (0.0,))
            goal_ang = self.goal_pose[2]
            goal_pos = np.array(
                [
                    256 - self._scale[1] * self._length * np.sin(goal_ang),
                    256 + self._scale[1] * self._length * np.cos(goal_ang),
                    0.0,
                ]
            )
            if self.obs_mode == "pc2":
                center_pos = np.array([256, 256, 0.0])
                return np.concatenate([agent_pos, goal_pos, center_pos])[None]
            else:
                return np.concatenate([agent_pos, goal_pos])[None]

    def reset(self):
        obs = super().reset()
        self.render_cache = None
        return obs

    def step(self, action, dummy_reward=False):
        if self.ac_mode == "rel":
            action = self.agent.position + action[:2]
        else:
            action = action[:2]
        self.render_cache = None
        return super().step(action, dummy_reward=dummy_reward)

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        if self.render_cache is not None:
            return self.render_cache

        # get image
        img = super()._render_frame(mode="rgb_array")
        h, w = img.shape[0], img.shape[1]

        # get a set of points that represent the current state
        pts = []
        for shape in self.block.shapes:
            verts = [self.block.local_to_world(v) for v in shape.get_vertices()]
            pts += [np.array([v.x, v.y]) for v in verts]
        pts = np.array(pts)
        assert len(pts) == 8

        result = {
            "images": np.array([img]),
            "depths": np.zeros_like([img]),
            "pc": np.concatenate([pts, np.zeros_like(pts[:, [0]])], axis=-1),
        }
        self.render_cache = result
        return result
