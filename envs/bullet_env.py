import os
import inspect, functools

import pybullet as pb

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

class BulletEnv(object):
    
    def __init__(self, time_step=1.0/240, gravity=(0,0,-9.8), render=False, recorder="", **kwargs):
        self.info = {"gravity": gravity, "time_step": time_step, "log": {}}
        self.recorder = recorder
        self.video_logger = None
        self._render = render
        self.bullet_cid = -1
        
    def __del__(self):
        self.close()

    def __getattr__(self, name):
        attr = getattr(pb, name)
        if inspect.isbuiltin(attr):
            attr = functools.partial(attr, physicsClientId=self.bullet_cid)
        return attr

    def connect(self, connect_mode=None, connect_options=""):
        kwargs = {}
        if connect_options: kwargs["options"] = connect_options
        if connect_mode:
            self.bullet_cid = pb.connect(connect_mode, **kwargs)
        else:
            self.bullet_cid = pb.connect(pb.GUI if self._render else pb.DIRECT, **kwargs)
        if self.recorder:
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=self.bullet_cid)
            self.video_logger = pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4, self.recorder,
                                                     physicsClientId=self.bullet_cid)

        self.reset_simulation()
        self.time_step = self.info["time_step"]
        self.gravity = self.info["gravity"]

    def render(self, mode="human"):
        self.close()
        self._render = True
        self.init()

    def reset_simulation(self, *args, **kwargs):
        pb.resetSimulation(*args, **kwargs, physicsClientId=self.bullet_cid)

    def do_simulation(self):
        pb.stepSimulation(physicsClientId=self.bullet_cid)
        
    def step(self):
        self.do_simulation()
    
    def close(self, bullet_cid=None):
        if bullet_cid is None and not hasattr(self, "bullet_cid"): return
        if bullet_cid is None: bullet_cid = self.bullet_cid
        if hasattr(self, "video_logger") and self.video_logger is not None:
            pb.stopStateLogging(self.video_logger, physicsClientId=bullet_cid)
        try:
            pb.disconnect(physicsClientId=bullet_cid)
        except pb.error:
            pass

    def seed(self, s):
        pass

    def load_plugin(self, *args, **kwargs):
        return pb.loadPlugin(*args, **kwargs, physicsClientId=self.bullet_cid)

    def save_state(self, *args, **kwargs):
        return pb.saveState(*args, **kwargs, physicsClientId=self.bullet_cid)

    def restore_state(self, *args, **kwargs):
        return pb.restoreState(*args, **kwargs, physicsClientId=self.bullet_cid)
        
    def remove_state(self, *args, **kwargs):
        return pb.removeState(*args, **kwargs, physicsClientId=self.bullet_cid)

    @property
    def connection_info(self):
        return pb.getConnectionInfo(physicsClientId=self.bullet_cid)
        
    @property
    def time_step(self):
        return self.info["time_step"]

    @time_step.setter
    def time_step(self, val):
        self.info["time_step"] = val
        if self.bullet_cid is not None:
            return pb.setTimeStep(val, physicsClientId=self.bullet_cid)

    @property
    def gravity(self):
        return self.info["gravity"]

    @gravity.setter
    def gravity(self, val):
        if self.bullet_cid is not None:
            pb.setGravity(val[0], val[1], val[2], physicsClientId=self.bullet_cid)
        self.info["gravity"] = val
        
    @property
    def physics_engine_parameters(self):
        return pb.getPhysicsEngineParameters(physicsClientId=self.bullet_cid)

    def set_collision_filter_group_mask(self, *args, **kwargs):
        return pb.setCollisionFilterGroupMask(*args, **kwargs, physicsClientId=self.bullet_cid)

    def load_urdf(self, *args, **kwargs):
        return pb.loadURDF(*args, **kwargs, physicsClientId=self.bullet_cid)

    def configure_debug_visualizer(self, *args, **kwargs):
        return pb.configureDebugVisualizer(*args, **kwargs, physicsClientId=self.bullet_cid)

    # def set_gravity(self, *args, **kwargs):
    #     return pb.setGravity(*args, **kwargs, physicsClientId=self.bullet_cid)

    def set_joint_motor_control2(self, *args, **kwargs):
        return pb.setJointMotorControl2(*args, **kwargs, physicsClientId=self.bullet_cid)

    def set_joint_motor_control_array(self, *args, **kwargs):
        return pb.setJointMotorControlArray(*args, **kwargs, physicsClientId=self.bullet_cid)

    def set_joint_motor_control_multi_dof(self, *args, **kwargs):
        return pb.setJointMotorControlMultiDof(*args, **kwargs, physicsClientId=self.bullet_cid)
        
    def set_joint_motor_control_multi_dof_array(self, *args, **kwargs):
        return pb.setJointMotorControlMultiDofArray(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_num_joints(self, *args, **kwargs):
        return pb.getNumJoints(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_joint_info(self, *args, **kwargs):
        return pb.getJointInfo(*args, **kwargs, physicsClientId=self.bullet_cid)

    # def get_joint_state(self, *args, **kwargs):
    #     return pb.getJointState(*args, **kwargs, physicsClientId=self.bullet_cid)

    # def reset_joint_state(self, *args, **kwargs):
    #     return pb.resetJointState(*args, **kwargs, physicsClientId=self.bullet_cid)

    # def get_joint_states(self, *args, **kwargs):
    #     return pb.getJointStates(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_joint_state_multi_dof(self, *args, **kwargs):
        return pb.getJointStateMultiDof(*args, **kwargs, physicsClientId=self.bullet_cid)

    def reset_joint_state_multi_dof(self, *args, **kwargs):
        return pb.resetJointStateMultiDof(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_joint_states_multi_dof(self, *args, **kwargs):
        return pb.getJointStatesMultiDof(*args, **kwargs, physicsClientId=self.bullet_cid)

    def reset_joint_states_multi_dof(self, *args, **kwargs):
        return pb.resetJointStatesMultiDof(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_link_state(self, *args, **kwargs):
        return pb.getLinkState(*args, **kwargs, physicsClientId=self.bullet_cid)
    
    def get_link_states(self, *args, **kwargs):
        return pb.getLinkStates(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_dynamics_info(self, *args, **kwargs):
        return pb.getDynamicsInfo(*args, **kwargs, physicsClientId=self.bullet_cid)
    
    def change_dynamics(self, *args, **kwargs):
        return pb.changeDynamics(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_aabb(self, *args, **kwargs):
        return pb.getAABB(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_base_position_and_orientation(self, *args, **kwargs):
        return pb.getBasePositionAndOrientation(*args, **kwargs, physicsClientId=self.bullet_cid)
    
    def reset_base_position_and_orientation(self, *args, **kwargs):
        # pybullet will reset base velocity to zero while reseting the base position and orientation
        i = kwargs["objectUniqueId"] if "objectUniqueId" in kwargs else args[0]
        v = pb.getBaseVelocity(i, physicsClientId=self.bullet_cid)
        res = pb.resetBasePositionAndOrientation(*args, **kwargs, physicsClientId=self.bullet_cid)
        pb.resetBaseVelocity(i, *v, physicsClientId=self.bullet_cid)
        return res
    
    def get_base_velocity(self, *args, **kwargs):
        return pb.getBaseVelocity(*args, **kwargs, physicsClientId=self.bullet_cid)

    def reset_base_velocity(self, *args, **kwargs):
        return pb.resetBaseVelocity(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_contact_points(self, *args, **kwargs):
        return pb.getContactPoints(*args, **kwargs, physicsClientId=self.bullet_cid)

    def calculate_mass_matrix(self, *args, **kwargs):
        return pb.calculateMassMatrix(*args, **kwargs, physicsClientId=self.bullet_cid)

    def calculate_inverse_dynamics(self, *args, **kwargs):
        return pb.calculateInverseDynamics(*args, **kwargs, physicsClientId=self.bullet_cid)

    def create_collision_shape(self, *args, **kwargs):
        return pb.createCollisionShape(*args, **kwargs, physicsClientId=self.bullet_cid)

    def remove_collision_shape(self, *args, **kwargs):
        return pb.removeCollisionShape(*args, **kwargs, physicsClientId=self.bullet_cid)

    def create_visual_shape(self, *args, **kwargs):
        return pb.createVisualShape(*args, **kwargs, physicsClientId=self.bullet_cid)
    
    def change_visual_shape(self, *args, **kwargs):
        return pb.changeVisualShape(*args, **kwargs, physicsClientId=self.bullet_cid)

    def create_multi_body(self, *args, **kwargs):
        return pb.createMultiBody(*args, **kwargs, physicsClientId=self.bullet_cid)

    def remove_body(self, *args, **kwargs):
        return pb.removeBody(*args, **kwargs, physicsClientId=self.bullet_cid)

    def reset_debug_visualizer_camera(self, *args, **kwargs):
        return pb.resetDebugVisualizerCamera(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_camera_image(self, *args, **kwargs):
        return pb.getCameraImage(*args, **kwargs, physicsClientId=self.bullet_cid)

    def enable_joint_force_torque_sensor(self, *args, **kwargs):
        return pb.enableJointForceTorqueSensor(*args, **kwargs, physicsClientId=self.bullet_cid)

