import os

import numpy as np

from . import bullet_env
from .utils import spd_controller

class Agent(object):

    def __init__(self, env):
        self.env = env
        self.control_param = None

        self.id = None
        self.control_param = {
            "targetPositions": None,
            "forces": None
        }
    
    def init(self, obj):
        self.id = obj
        self._n_joints = self.env.get_num_joints(self.id)

        self.joints = {}
        self.links = {}
        self.joint_name = []
        self.link_name = []
        self.motors = []
        self.torque_lim = []
        self.movement_lim = []
        self.end_effectors = list(range(self.n_joints))
        for j in range(self.n_joints):
            info = self.joint_info(j)
            joint_type = info[2]
            joint_name = info[1].decode("ascii")
            link_name = info[12].decode("ascii")
            self.joint_name.append(joint_name)
            self.link_name.append(link_name)
            parent = info[16]
            if joint_type == self.env.JOINT_REVOLUTE or \
               joint_type == self.env.JOINT_SPHERICAL:
                low_lim, high_lim, torque_lim = info[8:11]
                self.motors.append(j)
                self.torque_lim.append(torque_lim)
                self.movement_lim.append((low_lim, high_lim))
            else:
                assert(joint_type in [self.env.JOINT_FIXED, self.env.JOINT_REVOLUTE, self.env.JOINT_SPHERICAL])
            self.joints[joint_name] = j
            self.links[link_name] = j
            if parent in self.end_effectors:
                self.end_effectors.remove(parent)

    @property
    def pose(self):
        pose = {}
        pose["base_position"], pose["base_orientation"] = self.base_position_and_orientation
        pose["base_linear_velocity"], pose["base_angular_velocity"] = self.base_velocity
        for jid in self.motors:
            pose[jid] = {}
            pose[jid]["position"], pose[jid]["velocity"], *_ = self.joint_state(jid)
        return pose

    @pose.setter
    def pose(self, pose):
        # disable previous control signal using force = 0
        for jid in range (self.n_joints):
	        self.env.set_joint_motor_control2(self.id,
                jid, targetPosition=0, targetVelocity=0, force=0,
                controlMode=self.env.POSITION_CONTROL
            )
	        self.env.set_joint_motor_control_multi_dof(self.id,
                jid, targetPosition=[0,0,0,1], targetVelocity=[0,0,0], force=[0,0,0],
                controlMode=self.env.POSITION_CONTROL
            )
        self.env.reset_base_position_and_orientation(self.id,
            pose["base_position"], pose["base_orientation"]
        )
        self.env.reset_base_velocity(self.id,
            pose["base_linear_velocity"], pose["base_angular_velocity"]
        )
        for name, jid in self.joints.items():
            if jid in pose:
                key = jid
            elif name in pose:
                key = name
            else:
                continue
            if not hasattr(pose[key]["position"], "__len__"):
                self.env.reset_joint_state(self.id, jid,
                    pose[key]["position"], pose[key]["velocity"]
                )
            elif len(pose[key]["position"]) > 0:
                self.env.reset_joint_state_multi_dof(self.id, jid,
                    pose[key]["position"], pose[key]["velocity"]
                )

    @property
    def target_position(self):
        return self.control_param["targetPositions"]
    
    @target_position.setter
    def target_position(self, val):
        self.control_param["targetPositions"] = val
        self._update_target_position()
    
    @property
    def target_torque(self):
        return self.control_param["forces"]
    
    @target_torque.setter
    def target_torque(self, val):
        self.control_param["forces"] = val
        self._update_target_torque()
    
    def _update_target_position(self):
        pass
    
    def _update_target_torque(self):
        pass
    
    @property
    def n_joints(self):
        return self._n_joints
    
    @property
    def n_links(self):
        return self._n_joints

    def joint_info(self, jid):
        return self.env.get_joint_info(self.id, jid)
    
    def joint_state(self, jid):
        if hasattr(jid, "__len__"):
            return self.env.get_joint_states_multi_dof(self.id, jid)
        return self.env.get_joint_state_multi_dof(self.id, jid)

    def link_info(self, lid):
        return self.env.get_link_info(self.id, lid)
    
    def link_state(self, lid, compute_link_velocity=0, compute_forward_kinematics=0):
        if hasattr(lid, "__len__"):
            return self.env.get_link_states(self.id, lid, computeLinkVelocity=compute_link_velocity, computeForwardKinematics=compute_forward_kinematics)
        return self.env.get_link_state(self.id, lid, computeLinkVelocity=compute_link_velocity, computeForwardKinematics=compute_forward_kinematics)
    
    def dynamics_info(self, lid):
        return self.env.get_dynamics_info(self.id, lid)
    
    def change_dynamics(self, *args, **kwargs):
        return self.env.change_dynamics(self.id, *args, **kwargs)
    
    def change_visual_shape(self, *args, **kwargs):
        return self.env.change_visual_shape(self.id, *args, **kwargs)

    def aabb(self, lid):
        return self.env.get_aabb(self.id, lid)

    @property
    def base_position_and_orientation(self):
        return self.env.get_base_position_and_orientation(self.id)
    
    def get_base_position_and_orientation(self):
        return self.env.get_base_position_and_orientation(self.id)

    @base_position_and_orientation.setter
    def base_position_and_orientation(self, pos_orient):
        return self.env.reset_base_position_and_orientation(self.id, pos_orient[0], pos_orient[1])

    def reset_base_position_and_orientation(self, pos, orient):
        return self.env.reset_base_position_and_orientation(self.id, pos, orient)
    
    @property
    def base_velocity(self):
        return self.env.get_base_velocity(self.id)
    
    def get_base_velocity(self):
        return self.env.get_base_velocity(self.id)
    
    def has_contact(self, tar=None, exclusive_links=None, inclusive_links=[]):
        for p in self.env.get_contact_points():
            if p[1] == p[2]: continue # ignore self-collision
            if p[1] == self.id:
                if tar is not None and p[2] != tar:
                    continue
                contact_part = p[3]
            elif p[2] == self.id:
                if tar is not None and p[1] != tar:
                    continue
                contact_part = p[4]
            else: continue
            if contact_part in inclusive_links:
                return True
            if exclusive_links is not None and contact_part not in exclusive_links:
                return True
        return False

    def _update_target_position(self):
        #set_joint_motor_control_multi_dof_array does not support position control with spherical joints
        i = 0
        kp = 0.2
        for jid, torque_lim in zip(self.motors, self.torque_lim):
            info = self.joint_info(jid)
            joint_type = info[2]
            if joint_type == self.env.JOINT_REVOLUTE:
                q = self.control_param["targetPositions"][i]
                i += 1
                self.env.set_joint_motor_control2(self.id,
                    jid, targetPosition=q,
                    controlMode=self.env.POSITION_CONTROL, positionGain=kp,
                    force=torque_lim
                )
            else: # joint_type == self.env.JOINT_SPHERICAL
                q = (
                    self.control_param["targetPositions"][i],
                    self.control_param["targetPositions"][i+1],
                    self.control_param["targetPositions"][i+2],
                    self.control_param["targetPositions"][i+3]
                )
                i += 4
                self.env.set_joint_motor_control_multi_dof(self.id,
                    jid, targetPosition=q,
                    controlMode=self.env.POSITION_CONTROL, positionGain=kp,
                    force=[torque_lim]
                )

    def _update_target_torque(self):
        forces = []
        i = 0
        for jid, torque_lim in zip(self.motors, self.torque_lim):
            info = self.joint_info(jid)
            joint_type = info[2]
            if joint_type == self.env.JOINT_REVOLUTE:
                f = [min(torque_lim, max(-torque_lim, self.control_param["forces"][i]))]
                i += 1
            else: # joint_type == self.env.JOINT_SPHERICAL
                f = [self.control_param["forces"][i], self.control_param["forces"][i+1], self.control_param["forces"][i+2]]
                n = np.linalg.norm(f)
                if n > torque_lim:
                    a = torque_lim/n
                    f = [v*a for v in f]
                i += 3
            forces.append(f)
        self.env.set_joint_motor_control_multi_dof_array(
            self.id, self.motors, self.env.TORQUE_CONTROL,
            forces=forces
        )


class Humanoid(Agent):
    
    AGENT_FILE = os.path.join(bullet_env.DATA_DIR, "humanoid.y_up.urdf")
    UP_AXIS = 1
    
    # for SPD controller
    SPD_KP = {
        "abdomen": 1000,
        "neck": 100,
        "right_hip": 500,
        "right_knee": 500,
        "right_ankle": 400, 
        "right_shoulder": 400,
        "right_elbow": 300,
        "left_hip": 500, 
        "left_knee": 500,
        "left_ankle": 400, 
        "left_shoulder": 400,
        "left_elbow": 300,
    }
    SPD_KD = {
        "abdomen": 100,
        "neck": 10,
        "right_hip": 50,
        "right_knee": 50,
        "right_ankle": 40,
        "right_shoulder": 40,
        "right_elbow": 30,
        "left_hip": 50,
        "left_knee": 50,
        "left_ankle": 40,
        "left_shoulder": 40,
        "left_elbow": 30,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        model = self.env.load_urdf(self.AGENT_FILE, [0,0,0], globalScaling=0.25, useFixedBase=False,
            flags=self.env.URDF_MAINTAIN_LINK_ORDER +
                self.env.URDF_USE_SELF_COLLISION +
                self.env.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )
        super().init(model)

        self.joint_pairs, self.joint_groups = [], {}

        if "head" in self.links:
            self.end_effectors.remove(self.links["head"])
        if "neck" in self.links:
            self.end_effectors.remove(self.links["neck"])
        for _, lid in self.links.items():
            # it is bugged if changing friction with damping at the same time
            self.change_dynamics(lid, lateralFriction=0.9)
            self.change_dynamics(lid, linearDamping=0, angularDamping=0)
        self.change_dynamics(-1, lateralFriction=0.9)
        self.change_dynamics(-1, linearDamping=0, angularDamping=0)
        
        # for SPD controller
        # # self-implemented SPD
        # self.kp, self.kd = [], []
        # self.valid_index, i = [], 0
        # for jid in self.motors:
        #     info = self.joint_info(jid)
        #     joint_type = info[2]
        #     joint_name = info[1].decode("ascii")
        #     if joint_type == self.env.JOINT_REVOLUTE:
        #         self.kp.append(self.SPD_KP[joint_name])
        #         self.kd.append(self.SPD_KD[joint_name])
        #         self.valid_index.append(i)
        #         i += 1
        #     else: # joint_type == self.env.JOINT_SPHERICAL
        #         self.kp.extend([self.SPD_KP[joint_name], self.SPD_KP[joint_name], self.SPD_KP[joint_name], 0])
        #         self.kd.extend([self.SPD_KD[joint_name], self.SPD_KD[joint_name], self.SPD_KD[joint_name], 0])
        #         self.valid_index.append(i)
        #         self.valid_index.append(i+1)
        #         self.valid_index.append(i+2)
        #         i += 4

        # Pybullet SPD
        self.kp, self.kd = [], []
        self.max_forces, self.zero_vel = [], []
        for jid, tau in zip(self.motors, self.torque_lim):
            info = self.joint_info(jid)
            joint_type = info[2]
            joint_name = info[1].decode("ascii")
            self.kp.append(self.SPD_KP[joint_name])
            self.kd.append(self.SPD_KD[joint_name])
            if joint_type == self.env.JOINT_REVOLUTE:
                self.max_forces.append([tau])
                self.zero_vel.append([0.])
            else: # joint_type == self.env.JOINT_SPHERICAL
                self.max_forces.append([tau, tau, tau])
                self.zero_vel.append([0., 0., 0.])

    def _update_target_position(self):
        # # self-implemented SPD
        # self.target_torque = spd_controller(
        #     self.env, self.id,
        #     self.control_param["targetPositions"], [0]*len(self.control_param["targetPositions"]),
        #     self.kp, self.kd,
        #     self.env.time_step
        # )[self.valid_index]

        # Pybullet SPD
        p = []
        i = 0
        for jid, f in zip(self.motors, self.max_forces):
            if len(f) == 1:
                p.append([self.control_param["targetPositions"][i]])
                i += 1
            else: # len(f) == 3
                p.append([
                    self.control_param["targetPositions"][i],
                    self.control_param["targetPositions"][i+1],
                    self.control_param["targetPositions"][i+2],
                    self.control_param["targetPositions"][i+3]
                ])
                i += 4
        self.env.set_joint_motor_control_multi_dof_array(
            self.id, self.motors, self.env.STABLE_PD_CONTROL,
            targetPositions=p,
            targetVelocities=self.zero_vel,
            forces=self.max_forces,
            positionGains=self.kp, velocityGains=self.kd
        )

