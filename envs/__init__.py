import os, time
from collections import deque
from typing import Sequence

import numpy as np

from . import agent
from . import spaces
from .bullet_env import BulletEnv
from .utils import quatmultiply, orient2heading, quat2mat, euler_zyx2quat, axis_angle2quat

from .reference_motion import MOTION_DIR, ReferenceMotion

class Observation(object):
    def __init__(self, history=-1, with_velocity=True):
        self.history = history
        self.with_velocity = with_velocity

class MotionImitate(BulletEnv):
    AGENT = agent.Humanoid

    def __init__(self, action: Sequence[str], observation: (Observation or dict[str, Observation]), **kwargs):
        def arg_parse(name, def_val):
            return kwargs[name] if kwargs is not None and name in kwargs else def_val
        
        self.fps = arg_parse("fps", 30.0)
        self.frame_skip = arg_parse("frame_skip", 20)
        kwargs["time_step"] = arg_parse("time_step", 1.0/(self.fps*self.frame_skip))
        super().__init__(**kwargs)
        self.overtime = arg_parse("overtime", 500)
        self.random_init_pose = arg_parse("random_init_pose", True)
        self.observation = observation

        self.agent = self.AGENT(self)
        self.ref_motion = ReferenceMotion(self.agent, action, motion_dir=arg_parse("motion_dir", MOTION_DIR))
        self.ground = None

        self.init_sim_state = None
        self.rng = np.random.RandomState()
        self.init()

    def init(self):
        self.simulation_steps = 0
        self.elapsed_time = 0
        self.init_time = 0
        if self.init_sim_state is not None:
            self.remove_state(self.init_sim_state)
            self.init_sim_state = None
        super().connect()
                
        self.agent.init()
        self.up_dir = self.agent.UP_AXIS
        self.ground = self.build_ground()
        g = [0, 0, 0]
        g[self.up_dir] = -9.8
        self.gravity = g
        if self.up_dir == 1:
            self.configure_debug_visualizer(self.COV_ENABLE_Y_AXIS_UP,1)
        elif self.up_dir == 0:
            raise ValueError("Unsupported to use X-axis as the up axis.")

        self.init_sim_state = self.save_state()
        if self.ref_motion is not None:
            self.ref_motion.init()
            self.init_action_space()
            self.init_observation_space()
            
    def build_ground(self):
        import pybullet_data
        self.setAdditionalSearchPath(pybullet_data.getDataPath())
        if self.up_dir == 2:
            orient = [0, 0, 0, 1]
        elif self.up_dir == 1:
            orient = euler_zyx2quat(0, 0, -np.pi * 0.5)
        else:
            assert(self.up_dir in [1, 2])
        ground = self.load_urdf("plane_implicit.urdf", [0, 0, 0], orient,
            useMaximalCoordinates=True)
        self.change_dynamics(ground, -1, lateralFriction=0.9)
        return ground

    def init_observation_space(self):
        obs = self.observe(self.agent)
        if isinstance(obs, dict):
            self.observation_space = {} 
            for k, ob in obs.items():
                self.observation_space[k] = spaces.Box()
                self.observation_space[k].shape = np.array(ob).shape
        else:
            self.observation_space = spaces.Box()
            self.observation_space.shape = np.array(obs).shape

    def init_action_space(self):
        self.action_mean, self.action_std = [], []
        lower_bound, upper_bound = [], []
        control_range = 2 # control range for the position controller, 2 means 2 times of the oringal movement range
        for jid, lim in zip(self.agent.motors, self.agent.movement_lim):
            info = self.agent.joint_info(jid)
            joint_type = info[2]
            if joint_type == self.JOINT_REVOLUTE:
                self.action_mean.append(0.5*(lim[1] + lim[0]))
                self.action_std.append((lim[1] - lim[0])*control_range)
                lower_bound.append(-1.0)
                upper_bound.append(1.0)
            elif joint_type == self.JOINT_SPHERICAL:
                y_offset, z_offset = 0, 0.2 # self.up_dir == 1
                if self.up_dir == 2:
                    y_offset, z_offset = -z_offset, y_offset
                self.action_mean.extend([0, y_offset, z_offset, 0])  # in order of axis, angle
                self.action_std.extend([1, 1, 1, (lim[1] - lim[0])*control_range])
                lower_bound.extend([-1.0, -1.0-y_offset, -1.0-z_offset, -1.0])
                upper_bound.extend([1.0, 1.0-y_offset, 1.0-z_offset, 1.0])
            else:
                assert(joint_type in [self.JOINT_REVOLUTE, self.JOINT_SPHERICAL])
        self.action_space = spaces.Box()
        self.action_space.shape = [len(lower_bound)]
        self.action_space.low, self.action_space.high = lower_bound, upper_bound

    def pre_process_action(self, action):
        action = np.add(self.action_mean, np.multiply(action, self.action_std))        
        i = 0
        for jid in self.agent.motors:
            joint_type = self.agent.joint_info(jid)[2]
            if joint_type == self.JOINT_REVOLUTE:
                i += 1
            else: # joint_type == self.JOINT_SPHERICAL
                q = axis_angle2quat((action[i+0], action[i+1], action[i+2]), action[i+3])
                action[i+0], action[i+1], action[i+2], action[i+3] = q
                i += 4
        return action

    def seed(self, s):
        self.rng.seed(s)

    def reset(self):
        self.restore_state(self.init_sim_state)
        self.simulation_steps = 0

        if len(self.ref_motion) > 1:
            target_ref_motion = self.rng.randint(len(self.ref_motion.motions))
            self.ref_motion.set_motion(target_ref_motion)
        self.ref_motion.reset()
        self.contactable_links = None if self.ref_motion.contactable_links is None else [
            self.agent.links[l] for l in self.ref_motion.contactable_links
        ]

        if self.random_init_pose and self.ref_motion.random_init_pose:
            phase = self.rng.rand()
        else:
            phase = 0.0
        if self.ref_motion.loopable:
            self.init_time = phase * self.ref_motion.duration
        else:
            self.init_time = phase * max(0, self.ref_motion.duration - 100/self.fps)
     
        self.elapsed_time = self.init_time
        ref_pose = self.ref_motion.set_sim_time(self.elapsed_time, add_noise=True)
        self.agent.pose = ref_pose

        dist = np.inf
        for lid in range(self.agent.n_links):
            aabb_min, aabb_max = self.agent.aabb(lid)
            dist = min(dist, aabb_min[self.up_dir])
        dist -= 0.001
        if dist < 0:
            ref_pose["base_position"][self.up_dir] -= dist
            self.agent.base_position_and_orientation = ref_pose["base_position"], ref_pose["base_orientation"]
            self.ref_motion.sync(self.elapsed_time, ref_pose["base_position"], None)
            self.ref_motion.set_sim_time(self.elapsed_time)

        state = self.observe(self.agent)
        self.info["expert_state"] = self.observe(self.ref_motion.agent)
        if self._render: self._render_delay = time.time()
        return state
    
    def step(self, action):
        action = self.pre_process_action(action)
        assert(not np.any(np.isnan(action)))

        self.simulation_steps += 1
        phase = self.ref_motion.phase(self.elapsed_time)

        for frame in range(self.frame_skip):
            self.elapsed_time += self.time_step

            self.agent.target_position = action
            self.do_simulation()
                
            if self._render:
                t = time.time()
                time.sleep(max(0, self.time_step - (t-self._render_delay)/1000))
                pos, orient = self.agent.base_position_and_orientation
                self.reset_debug_visualizer_camera(2, 180, 0, [pos[0], 0.84, pos[2]])
                self._render_delay = t
            
        if self.ref_motion.loopable and self.ref_motion.phase(self.elapsed_time) < phase:
            self.ref_motion.accumulate_cycle_offset()
        self.ref_motion.set_sim_time(self.elapsed_time)

        if self.contactable_links is None:
            fail = False
        else:
            fail = self.agent.has_contact(self.ground, exclusive_links=self.contactable_links)
        terminal = fail
        if not terminal and self.ref_motion and not self.ref_motion.loopable:
            terminal = self.elapsed_time >= self.ref_motion.duration
        if not terminal and self.overtime:
            self.info["TimeLimit.truncated"] = self.simulation_steps >= self.overtime
            terminal = self.info["TimeLimit.truncated"]
        reward = self.reward(fail)
        state = self.observe(self.agent)
        self.info["expert_state"] = self.observe(self.ref_motion.agent)
        
        return state, reward, terminal, self.info

    def observe(self, agent):
        base_pos, base_orient = agent.base_position_and_orientation
        base_height = base_pos[self.up_dir]

        up_dir_vec = [0, 0, 0]
        up_dir_vec[self.up_dir] = 1
        root_p_inv = [-p for p in base_pos]
        heading = orient2heading(base_orient, self.up_dir)
        root_q_inv = axis_angle2quat(up_dir_vec, -heading)
        root_R_inv = quat2mat(root_q_inv)

        obs = {}
        for key, opt in self.observation.items():
            with_velocity = opt.with_velocity
            if not hasattr(agent, "state_history"):
                agent.state_history = {}
            if key not in agent.state_history:
                agent.state_history[key] = deque(maxlen=opt.history)
            buffer = agent.state_history[key]
            if self.simulation_steps == 0:
                buffer.clear()
        
            joints = list(range(agent.n_links))
            if with_velocity:
                full_states = [
                    (s[0], s[1], s[6], s[7]) # p, q, v, w
                    for s in agent.link_state(joints,
                    compute_forward_kinematics=True, compute_link_velocity=True)
                ]
                buffer.append(full_states)
                s = []
                for _ in buffer:
                    for p, q, v, w in _:
                        p = np.matmul(root_R_inv, np.add(p, root_p_inv))
                        q = quatmultiply(root_q_inv, q)
                        if q[3] < 0: q = [-e for e in q]
                        s.extend(p)
                        s.extend(q)
                        v = np.matmul(root_R_inv, v)
                        w = np.matmul(root_R_inv, w)
                        s.extend(v)
                        s.extend(w)
            else:
                full_states = [
                    (s[0], s[1]) # p, q
                    for s in agent.link_state(joints,
                    compute_forward_kinematics=True)
                ]
                buffer.append(full_states)
                s = []
                for _ in buffer:
                    for p, q in _:
                        p = np.matmul(root_R_inv, np.add(p, root_p_inv))
                        q = quatmultiply(root_q_inv, q)
                        if q[3] < 0: q = [-e for e in q]
                        s.extend(p)
                        s.extend(q)

            s = np.array(s, dtype=np.float32)
            s.shape = (len(buffer), -1)
            obs[key] = s
        return next(iter(obs.values())) if len(obs) == 1 else obs

    def reward(self, terminal):
        if terminal:
            reward = 0
        else:
            reward = 1
        return reward
