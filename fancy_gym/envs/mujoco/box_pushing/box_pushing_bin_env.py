import os

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv
import mujoco
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import (
    rot_to_quat,
    get_quaternion_error,
    q_max, q_min,
    q_dot_max, q_torque_max,
)

MAX_EPISODE_STEPS_BOX_PUSHING = 100

BOX_POS_BOUND = np.array([[0.3, -0.45, -0.01], [0.6, 0.45, -0.01]])

class BoxPushingEnvBase(MujocoEnv, utils.EzPickle):
    """
    franka box pushing environment
    action space:
        normalized joints torque * 7 , range [-1, 1]
    observation space:
    """

    def __init__(
        self,
        num_boxes: int = 10,
        frame_skip: int = 10,
        width: int = 244,
        height: int = 244,
    ):
        assert num_boxes <= MAX_NUM_BOXES
        utils.EzPickle.__init__(**locals())
        self._steps = 0
        self.frame_skip = frame_skip
        self.num_boxes = num_boxes
        self._q_max, self._q_min, self._q_dot_max = q_max, q_min, q_dot_max
        self.width, self.height = width, height

        self.init_qpos_box_pushing = np.zeros(7 + MAX_NUM_BOXES * 7)
        self.init_qvel_box_pushing = np.zeros(7 + MAX_NUM_BOXES * 6)
        self.boxes = ["box_" + str(i) for i in range(num_boxes)]
        self.joints = ["box_joint_" + str(i) for i in range(num_boxes)]
        self.hidden = ["box_joint_" + str(i) for i in range(num_boxes, MAX_NUM_BOXES)]

        # noise of 8deg ~ pi/21rad, ensure >95% values inside the range with 3sigma=range
        self.noisy_start_pos = lambda : np.clip(
            START_POS + np.random.normal(0, np.pi / 21 / 3, START_POS.shape),
            self._q_min,
            self._q_max
        )

        self._episode_energy = 0.
        MujocoEnv.__init__(
            self,
            model_path=os.path.join(
                os.path.dirname(__file__),
                "assets/box_pushing_bins.xml",
            ),
            frame_skip=self.frame_skip,
            mujoco_bindings="mujoco"
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))

        # Boxes that are extra are rendered inside the bins
        for i, joint in enumerate(self.hidden):
            x, y, z = i % 18 // 6, i % 6, i // 18  # arange in a grid
            start_idx = 7 * (1 + i + self.num_boxes)
            self.init_qpos_box_pushing[start_idx:start_idx + 7] = np.array(
                [1.21 + x * 0.12, -0.58 + 0.23 * y , 0.2 + z * 0.12, 0, 0, 0, 0]
            )

        # camera calibration utilities
        fovys = [self.model.cam("rgbd").fovy[0], self.model.cam("rgbd_cage").fovy[0]]
        focal = 0.5 * self.height / np.tan(np.array(fovys) * np.pi / 360)
        self.focal_mat = {
            "rgbd": np.array([
                [-focal[0], 0, self.width / 2.0, 0],
                [0, focal[0], self.height / 2.0, 0],
                [0, 0, 1, 0]
            ]),
            "rgbd_cage": np.array([
                [-focal[1], 0, self.width / 2.0, 0],
                [0, focal[1], self.height / 2.0, 0],
                [0, 0, 1, 0]
            ]),
        }

        self.near = self.model.vis.map.znear * self.model.stat.extent
        self.far = self.model.vis.map.zfar * self.model.stat.extent

    def step(self, action):
        action = 10 * np.clip(action, self.action_space.low, self.action_space.high)
        resultant_action = np.clip(action + self.data.qfrc_bias[:7].copy(), -q_torque_max, q_torque_max)

        unstable_simulation = False

        try:
            self.do_simulation(resultant_action, self.frame_skip)
        except Exception as e:
            print(e)
            unstable_simulation = True

        self._steps += 1
        self._episode_energy += np.sum(np.square(action))

        episode_end = True if self._steps >= MAX_EPISODE_STEPS_BOX_PUSHING else False

        box_pos = self.data.body("box_0").xpos.copy()
        box_quat = self.data.body("box_0").xquat.copy()
        rod_tip_pos = self.data.site("rod_tip").xpos.copy()
        rod_quat = self.data.body("push_rod").xquat.copy()
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()

        if not unstable_simulation:
            reward = self._get_reward(action, qpos, qvel, box_pos_xyz)
        else:
            reward = -50

        obs = self._get_obs()
        infos = {
            'episode_end': episode_end,
            'box_goal_pos_dist': box_goal_pos_dist,
            'episode_energy': 0. if not episode_end else self._episode_energy,
            'num_steps': self._steps
        }
        return obs, reward, episode_end, infos

    def reset_model(self):
        # rest box to initial position
        self.set_state(self.init_qpos_box_pushing, self.init_qvel_box_pushing)

        # Initialize box positions randomly, ensure collision free init by trial
        positions = []
        for joint in self.joints:
            new_pos, collision = self.sample_context(), True
            while collision and len(positions) > 0:
                for p in positions:
                    collision = np.linalg.norm(new_pos[:3] - p[:3]) < 0.15
                    if collision:  # collision detected sample new pos and check again
                        new_pos = self.sample_context()
                        break
            self.data.joint(joint).qpos = new_pos
            positions.append(new_pos)

        # Robot out of the way of boxes before dropping them
        self.data.qpos[:7] = START_POS + np.array([0, 0, 0, np.pi*5/8, 0, 0, 0])

        mujoco.mj_forward(self.model, self.data)
        self._steps, self._episode_energy = 0, 0

        # Init environemnt by letting boxes fall
        no_action = np.clip(
            np.zeros(self.action_space.shape) + self.data.qfrc_bias[:7].copy(),
            -q_torque_max,
            q_torque_max
        )
        self.do_simulation(no_action, BOX_INIT_FRAME_SKIPS)

        self.reset_robot_pos()

        return self._get_obs()

    def reset_robot_pos(self):
        self.data.qpos[:7] = self.noisy_start_pos()
        self.data.qvel[:7] = np.zeros(START_POS.shape)
        mujoco.mj_forward(self.model, self.data)

    def sample_context(self):
        pos = self.np_random.uniform(low=BOX_POS_BOUND[0], high=BOX_POS_BOUND[1])
        theta = self.np_random.uniform(low=0, high=np.pi * 2)
        quat = rot_to_quat(theta, np.array([0, 0, 1]))
        return np.concatenate([pos, quat])

    def _get_reward(self, action, qpos, qvel, box_pos=None):
        """
        By default the environment should learn smooth movement with the least torque
        necessary without violating constraints.

        Args:
            action (np.array): action taken during step
            qpos (np.array): robot position, angle for each joint
            qvel (np.array): robot velocity, torque for each joint
        Return:
            (float): scalar reward value
        """
        joint_penalty_reward = self._joint_limit_violate_penalty(
            qpos,
            qvel,
            enable_pos_limit=True,
            enable_vel_limit=True
        )
        energy_cost = -0.0005 * np.sum(np.square(action))

        return joint_penalty_reward + energy_cost

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos[:7].copy(),  # joint position
            self.data.qvel[:7].copy(),  # joint velocity
            # self.data.qfrc_bias[:7].copy(),  # joint gravity compensation
            # self.data.site("rod_tip").xpos.copy(),  # position of rod tip
            # self.data.body("push_rod").xquat.copy(),  # orientation of rod
            self.data.body("box_0").xpos.copy(),  # position of box
            self.data.body("box_0").xquat.copy(),  # orientation of box
        ])
        return obs

    def _joint_limit_violate_penalty(self, qpos, qvel, enable_pos_limit=False, enable_vel_limit=False):
        penalty = 0.
        p_coeff = 1.
        v_coeff = 1.
        # q_limit
        if enable_pos_limit:
            higher_error = qpos - self._q_max
            lower_error = self._q_min - qpos
            penalty -= p_coeff * (abs(np.sum(higher_error[qpos > self._q_max])) +
                                  abs(np.sum(lower_error[qpos < self._q_min])))
        # q_dot_limit
        if enable_vel_limit:
            q_dot_error = abs(qvel) - abs(self._q_dot_max)
            penalty -= v_coeff * abs(np.sum(q_dot_error[q_dot_error > 0.]))
        return penalty

    def get_body_jacp(self, name):
        id = mujoco.mj_name2id(self.model, 1, name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, None, id)
        return jacp

    def get_body_jacr(self, name):
        id = mujoco.mj_name2id(self.model, 1, name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, None, jacr, id)
        return jacr

    def calculateOfflineIK(self, desired_cart_pos, desired_cart_quat):
        """
        calculate offline inverse kinematics for franka pandas
        :param desired_cart_pos: desired cartesian position of tool center point
        :param desired_cart_quat: desired cartesian quaternion of tool center point
        :return: joint angles
        """
        J_reg = 1e-6
        w = np.diag([1, 1, 1, 1, 1, 1, 1])
        target_theta_null = np.array([
            3.57795216e-09,
            1.74532920e-01,
            3.30500960e-08,
            -8.72664630e-01,
            -1.14096181e-07,
            1.22173047e00,
            7.85398126e-01])
        eps = 1e-5          # threshold for convergence
        IT_MAX = 1000
        dt = 1e-3
        i = 0
        pgain = [
            33.9403713446798,
            30.9403713446798,
            33.9403713446798,
            27.69370238555632,
            33.98706171459314,
            30.9185531893281,
        ]
        pgain_null = 5 * np.array([
            7.675519770796831,
            2.676935478437176,
            8.539040163444975,
            1.270446361314313,
            8.87752182480855,
            2.186782233762969,
            4.414432577659688,
        ])
        pgain_limit = 20
        q = self.data.qpos[:7].copy()
        qd_d = np.zeros(q.shape)
        old_err_norm = np.inf

        while True:
            q_old = q
            q = q + dt * qd_d
            q = np.clip(q, q_min, q_max)
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)
            current_cart_pos = self.data.body("tcp").xpos.copy()
            current_cart_quat = self.data.body("tcp").xquat.copy()

            cart_pos_error = np.clip(desired_cart_pos - current_cart_pos, -0.1, 0.1)

            if np.linalg.norm(current_cart_quat - desired_cart_quat) > np.linalg.norm(current_cart_quat + desired_cart_quat):
                current_cart_quat = -current_cart_quat
            cart_quat_error = np.clip(get_quaternion_error(current_cart_quat, desired_cart_quat), -0.5, 0.5)

            err = np.hstack((cart_pos_error, cart_quat_error))
            err_norm = np.sum(cart_pos_error**2) + np.sum((current_cart_quat - desired_cart_quat)**2)
            if err_norm > old_err_norm:
                q = q_old
                dt = 0.7 * dt
                continue
            else:
                dt = 1.025 * dt

            if err_norm < eps:
                break
            if i > IT_MAX:
                break

            old_err_norm = err_norm

            ### get Jacobian by mujoco
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)

            jacp = self.get_body_jacp("tcp")[:, :7].copy()
            jacr = self.get_body_jacr("tcp")[:, :7].copy()

            J = np.concatenate((jacp, jacr), axis=0)

            Jw = J.dot(w)

            # J * W * J.T + J_reg * I
            JwJ_reg = Jw.dot(J.T) + J_reg * np.eye(J.shape[0])

            # Null space velocity, points to home position
            qd_null = pgain_null * (target_theta_null - q)

            margin_to_limit = 0.1
            qd_null_limit = np.zeros(qd_null.shape)
            qd_null_limit_max = pgain_limit * (q_max - margin_to_limit - q)
            qd_null_limit_min = pgain_limit * (q_min + margin_to_limit - q)
            qd_null_limit[q > q_max - margin_to_limit] += qd_null_limit_max[q > q_max - margin_to_limit]
            qd_null_limit[q < q_min + margin_to_limit] += qd_null_limit_min[q < q_min + margin_to_limit]
            qd_null += qd_null_limit

            # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null
            qd_d = np.linalg.solve(JwJ_reg, pgain * err - J.dot(qd_null))

            qd_d = w.dot(J.transpose()).dot(qd_d) + qd_null

            i += 1

        return q
