from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    # Random x goal + random init pos
    @property
    def context_mask(self):
        return np.hstack([
            [False] * 7,  # joints position
            [False] * 7,  # joints velocity
            [False] * 3,  # position of box
            [False] * 4,  # orientation of box
            [True] * 3,  # position of target
            [True] * 4,  # orientation of target
            # [True] * 1,  # time
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos[:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[:7].copy()


class MPWrapperBin(MPWrapper):

    # Random x goal + random init pos
    @property
    def context_mask(self):
        return np.hstack([
            [True] * 7,  # joints position
            # [True] * 7,  # joints velocity
            # [False] * 7,  # joint gravity compensation
            # [True] * 3,  # position of tcp
            # [True] * 4,  # orientation of end effector
            [True] * 3 * self.num_boxes,  # position of boxes
            [True] * 1 * self.num_boxes,  # orientation of boxes
            # [True] * 1,  # time
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos[:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[:7].copy()
