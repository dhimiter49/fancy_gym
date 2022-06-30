from typing import Tuple, Union

import numpy as np

from alr_envs.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    @property
    def context_mask(self) -> np.ndarray:
        # Besides the ball position, the environment is always set to 0.
        return np.hstack([
            [False] * 2,  # cup position
            [True] * 2,  # ball position
            [False] * 2,  # cup velocity
            [False] * 2,  # ball velocity
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray]:
        return np.hstack([self.env.physics.named.data.qpos['cup_x'], self.env.physics.named.data.qpos['cup_z']])

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return np.hstack([self.env.physics.named.data.qvel['cup_x'], self.env.physics.named.data.qvel['cup_z']])

    @property
    def goal_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        raise ValueError("Goal position is not available and has to be learnt based on the environment.")

    @property
    def dt(self) -> Union[float, int]:
        return self.env.dt
