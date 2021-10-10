import os
import glob
import json
import math
from posixpath import relpath
import numpy as np
from copy import deepcopy

from . import bullet_env
from .utils import so_fb_butter_lpf
from .utils import quatmultiply, quatdiff, quatdiff_rel
from .utils import quat2axis_angle, axis_angle2quat
from .utils import lerp, slerp, rotate_vector, orient2heading

MOTION_DIR = os.path.join(bullet_env.DATA_DIR, "motions")

class Motion(object):
    def __init__(self):
        self.contactable_links = None
        self.loopable = False
        self.duration = 0
        self.frames = None
        self.cycle_pos_offset = None
        self.cycle_orient_offset = None
        self.sync_cycle_orient = False
        self.random_init_pose = False

class ReferenceMotion(object):

    def __init__(self, agent, action, motion_dir=MOTION_DIR):
        self.agent = agent
        if type(action) == str:
            self.motion_files = glob.glob(os.path.join(motion_dir, "{}.json".format(action)))
        else:
            self.motion_files = [p for act in action for p in glob.glob(os.path.join(motion_dir, "{}.json".format(act))) ]
        self.motions = []
        self.env = None
        
    def init(self):
        self.motions.clear()
        self.env = deepcopy(self.agent.env)
        self.env._render = False
        self.env.init_sim_state = None
        self.env.ref_motion = None
        self.env.init()
        self.agent = self.env.agent

        for j in range(-1, self.agent.n_joints):
            self.env.set_collision_filter_group_mask(self.agent.id, j, collisionFilterGroup=0, collisionFilterMask=0)
            self.agent.change_dynamics(j,
                activationState=self.env.ACTIVATION_STATE_SLEEP +
                    self.env.ACTIVATION_STATE_ENABLE_SLEEPING +
                    self.env.ACTIVATION_STATE_DISABLE_WAKEUP
            )

        def linear_vel(p0, p1, delta_t):
            if hasattr(p0, "__len__"):
                return [(v1-v0)/delta_t for v0, v1 in zip(p0, p1)]
            return (p1-p0)/delta_t

        def angular_vel(q0, q1, delta_t):
            axis, angle = quat2axis_angle(quatdiff(q0, q1))
            angle /= delta_t
            return [angle*a for a in axis]

        def angular_vel_rel(q0, q1, delta_t):
            axis, angle = quat2axis_angle(quatdiff_rel(q0, q1))
            angle /= delta_t
            return [angle*a for a in axis]

        for motion_file in self.motion_files:
            m = Motion()
            with open(motion_file) as f:
                data = json.load(f)
            frames = data["frames"]
            m.loopable = data["loopable"]
            if m.loopable < 0:
                m.loopable = False
            elif m.loopable == 0:
                m.loopable = True
            m.contactable_links = data["contactable_links"]
            if not m.contactable_links: m.contactable_links = None
            m.sync_cycle_orient = data["sync_cycle_orient"]
            m.random_init_pose = data["random_init_pose"]
            dt = data["sampling_interval"]
            m.duration = 0 
            m.frames = []
            for n in range(len(frames)-1):
                f = frames[n]
                f_ = frames[n+1]
                pose = {"time": m.duration}
                for name in f.keys():
                    if name == "base_position":
                        pose["base_position"] = f[name]
                        pose["base_linear_velocity"] = linear_vel(f[name], f_[name], dt)
                    elif name == "base_orientation":
                        pose["base_orientation"] = f[name]
                        pose["base_angular_velocity"] = angular_vel(f[name], f_[name], dt)    # dq q0 = q1, fixed frame
                    else:
                        pose[name] = {"position": f[name]}
                        assert(hasattr(f[name], "__len__") and (len(f[name]) == 4 or len(f[name]) == 1))
                        if len(f[name]) == 4:
                            pose[name]["velocity"] = angular_vel_rel(f[name], f_[name], dt)   # q0 dq = q1, related frame
                        else:
                            pose[name]["velocity"] = linear_vel(f[name], f_[name], dt)
                m.frames.append(pose)
                m.duration += dt
            f = frames[-1]
            pose = {
                "time": m.duration,
                "base_linear_velocity": [v for v in m.frames[-1]["base_linear_velocity"]],
                "base_angular_velocity": [v for v in m.frames[-1]["base_angular_velocity"]],
            }
            for name in f.keys():
                if name == "base_position":
                    pose["base_position"] = f[name]
                elif name == "base_orientation":
                    pose["base_orientation"] = f[name]
                else:
                    pose[name] = {
                        "position": f[name],
                        "velocity": [v for v in m.frames[-1][name]["velocity"]]
                    }
            m.frames.append(pose)
            
            fs = 1.0 / m.frames[1]["time"]
            fc = 6.0
            if fc*2 < fs:
                for i in range(3):
                    vel = so_fb_butter_lpf([p["base_linear_velocity"][i] for p in m.frames], fs, fc)
                    for v, p in zip(vel, m.frames): p["base_linear_velocity"][i] = v
                    vel = so_fb_butter_lpf([p["base_angular_velocity"][i] for p in m.frames], fs, fc)
                    for v, p in zip(vel, m.frames): p["base_angular_velocity"][i] = v
                for name in m.frames[0].keys():
                    if type(m.frames[0][name]) == dict:
                        for i in range(len(m.frames[0][name]["velocity"])):
                            vel = so_fb_butter_lpf([p[name]["velocity"][i] for p in m.frames], fs, fc)
                            for v, p in zip(vel, m.frames): p[name]["velocity"][i] = v
            if type(m.loopable) is bool:
                f0 = 0
            else:
                f0 = m.loopable
            m.cycle_pos_offset = [p_-p for p_, p in zip(m.frames[-1]["base_position"], m.frames[f0]["base_position"])]
            m.cycle_pos_offset[self.env.up_dir] = 0
            ref_dir = [0, 0, 0]
            ref_dir[self.env.up_dir] = 1
            heading_ = orient2heading(m.frames[-1]["base_orientation"], self.env.up_dir)
            heading  = orient2heading(m.frames[f0]["base_orientation"], self.env.up_dir)
            if abs(heading_-heading) > 0.5: # near 30 deg
                m.cycle_orient_offset = axis_angle2quat(ref_dir, heading_-heading)
            else:
                m.cycle_orient_offset = None

            self.motions.append(m)

        self.set_motion(0)
        self.base_pos_offset = [0, 0, 0]
        self.base_orient_offset = [0, 0, 0, 1] 

    def __len__(self):
        return len(self.motions)
    
    def __bool__(self):
        return bool(self.motions)
    
    def set_motion(self, idx):
        m = self.motions[idx]
        self.contactable_links = m.contactable_links
        self.loopable = m.loopable
        self.duration = m.duration
        self.frames = m.frames
        self.cycle_pos_offset = m.cycle_pos_offset
        self.cycle_orient_offset = m.cycle_orient_offset
        self.sync_cycle_orient = m.sync_cycle_orient
        self.random_init_pose = m.random_init_pose
        self.target_ref_motion = idx
        
    def set_sim_time(self, time, with_base_offset=True, add_noise=False):
        curr_pose = self.dummy_pose(time, with_base_offset, add_noise=add_noise)
        self.agent.pose = curr_pose
        return curr_pose

    def reset(self):
        self.set_base_pos_offset([0, 0, 0])
        self.set_base_orient_offset([0, 0, 0, 1])

    def set_base_pos_offset(self, offset):
        self.base_pos_offset = [v for v in offset]
    
    def set_base_orient_offset(self, offset):
        self.base_orient_offset = [v for v in offset]

    def phase(self, time):
        if self.loopable is False:
            if time > self.duration:
                f = 1.
            else:
                f = time / self.duration
        elif self.loopable is True:
            f = math.fmod(time / self.duration, 1.)
        else:
            start_time = self.frames[self.loopable]["time"]
            if time > start_time:
                time = math.fmod(time - start_time, self.duration - start_time) + start_time
            f = time / self.duration
        return self.target_ref_motion*2 + f

        
    def sync(self, time, tar_position=None, tar_orientation=None):
        ref_pose = self.dummy_pose(time, with_base_offset=False)
        if tar_position is not None:
            offset = [v-v0 for v, v0 in zip(tar_position, ref_pose["base_position"])]
            offset[self.env.up_dir] = 0
            self.set_base_pos_offset(offset)
        if tar_orientation is not None:
            heading = self.env.orient2heading(tar_orientation)
            ref_heading = self.env.orient2heading(ref_pose["base_orientation"])
            ref_dir = [0, 0, 0]
            ref_dir[self.env.up_dir] = 1
            offset = axis_angle2quat(ref_dir, heading-ref_heading)
            self.set_base_orient_offset(offset)  

    def accumulate_cycle_offset(self):
        if self.sync_cycle_orient and self.cycle_orient_offset:
            pos_offset = rotate_vector(self.cycle_pos_offset, self.base_orient_offset)
            pos_offset = np.add(self.base_pos_offset, pos_offset)
            self.set_base_pos_offset(pos_offset)
            orient_offset = quatmultiply(self.base_orient_offset, self.cycle_orient_offset)
            self.set_base_orient_offset(orient_offset)
        else:
            pos_offset = np.add(self.base_pos_offset, self.cycle_pos_offset)
            self.set_base_pos_offset(pos_offset)

    def noise(self, size=1):
        noise_std = 0.2
        noise_bound = 0.6
        noises = []
        while len(noises) < size:
            noise = self.env.rng.randn()*noise_std
            if noise >= -noise_bound and noise <= noise_bound:
                noises.append(noise)
        if size == 1:
            return noises[0]
        else:
            return noises
        
    def lerp_frame(self, f0, f1, frac, add_noise=False):
        if frac == 0:
            pose = {
                "base_position": f0["base_position"],
                "base_orientation": f0["base_orientation"],
                "base_linear_velocity": f0["base_linear_velocity"],
                "base_angular_velocity": f0["base_angular_velocity"]
            }
        else:
            pose = {
                "base_position": lerp(f0["base_position"], f1["base_position"], frac),
                "base_orientation": slerp(f0["base_orientation"], f1["base_orientation"], frac),
                "base_linear_velocity": lerp(f0["base_linear_velocity"], f1["base_linear_velocity"], frac),
                "base_angular_velocity": lerp(f0["base_angular_velocity"], f1["base_angular_velocity"], frac)
            }
        for name in f0.keys():
            if name not in self.agent.joints: continue
            jid = self.agent.joints[name]
            if frac == 0:
                p = f0[name]["position"]
                v = f0[name]["velocity"]
            else:
                if len(f0[name]["position"]) == 4:
                    p = slerp(f0[name]["position"], f1[name]["position"], frac)
                else:
                    p = lerp(f0[name]["position"], f1[name]["position"], frac)
                v = lerp(f0[name]["velocity"], f1[name]["velocity"], frac)
            if add_noise:
                if len(p) == 4:
                    noise_angle = self.noise()
                    noise_axis = np.random.uniform(-1, 1, size=3)
                    length = np.linalg.norm(noise_axis)
                    while length < np.finfo(float).eps:
                        noise_axis = np.random.uniform(-1, 1, size=3)
                        length = np.linalg.norm(noise_axis)
                    noise_axis = noise_axis / length
                    noise = axis_angle2quat(noise_axis, noise_angle)
                    p = quatmultiply(p, noise)
                else:
                    p = np.add(p, self.noise(len(p)))
                v = np.add(v, self.noise(len(v)))
            pose[jid] = {"position": p, "velocity": v}
        return pose 

    def dummy_pose(self, time, with_base_offset=True, add_noise=False):
        if self.loopable is False:
            if time > self.duration:
                time = self.duration
        elif self.loopable is True:
            time = math.fmod(time, self.duration)
        else:
            start_time = self.frames[self.loopable]["time"]
            if time > start_time:
                time = math.fmod(time - start_time, self.duration - start_time) + start_time
            
        # get two ref frames covering the specified time
        f0 = None
        for i in range(len(self.frames)):
            if self.frames[i]["time"] <= time:
                f0 = i
            if self.frames[i]["time"] > time:
                break
        f1 = f0 + 1
        if f1 > len(self.frames) - 1:
            f1 = f0 
        f0 = self.frames[f0]
        f1 = self.frames[f1]
        # lerp two frames
        dt = f1["time"] - f0["time"]
        if dt == 0: frac = 0
        else: frac = (time - f0["time"]) / dt
        pose = self.lerp_frame(f0, f1, frac, add_noise)
        return self._post_process(pose, with_base_offset)

    def _post_process(self, pose, with_base_offset):
        # offset the pose
        if with_base_offset:
            if self.base_orient_offset[0] != 0 or self.base_orient_offset[1] != 0 or self.base_orient_offset[2] != 0 or self.base_orient_offset[3] != 1:
                pose["base_orientation"] = quatmultiply(self.base_orient_offset, pose["base_orientation"])
                pose["base_linear_velocity"] = rotate_vector(pose["base_linear_velocity"], self.base_orient_offset)
                pose["base_angular_velocity"] = rotate_vector(pose["base_angular_velocity"], self.base_orient_offset)
                base_pos = np.subtract(pose["base_position"], self.frames[0]["base_position"])
                base_pos = rotate_vector(base_pos, self.base_orient_offset)
                base_pos = np.add(base_pos, self.base_pos_offset)
                pose["base_position"] = np.add(base_pos, self.frames[0]["base_position"])
            else:
                pose["base_position"] = np.add(pose["base_position"], self.base_pos_offset)
        return pose
