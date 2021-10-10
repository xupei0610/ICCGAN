import numpy as np
import pybullet as pb

def quatmultiply(q0, q1):
    x1, y1, z1, w1 = q0
    x2, y2, z2, w2 = q1
    return [
        w2*x1+x2*w1-y2*z1+z2*y1,
        w2*y1+x2*z1+y2*w1-z2*x1,
        w2*z1-x2*y1+y2*x1+z2*w1,
        w2*w1-x2*x1-y2*y1-z2*z1
    ]

def quatconj(q):
    return [-q[0], -q[1], -q[2], q[3]]

def quatdiff(q0, q1):
    return pb.getDifferenceQuaternion(q0, q1)

def quatdiff_rel(q0, q1):
    return quatmultiply(quatconj(q0), q1)

def vel2quat_diff(v, q):
	return np.matmul([
		[ 0.5 * q[3], -0.5 * q[2],  0.5 * q[1]], # x
		[ 0.5 * q[2],  0.5 * q[3], -0.5 * q[0]], # y
		[-0.5 * q[1],  0.5 * q[0],  0.5 * q[3]], # z
        [-0.5 * q[0], -0.5 * q[1], -0.5 * q[2]]  # w
    ], v)

def quat2mat(q):
    return np.reshape(pb.getMatrixFromQuaternion(q), (3,3))

def quat2euler_zyx(q):
    return pb.getEulerFromQuaternion(q)

def euler_zyx2quat(yaw, pitch, roll):
    # x first
    cy = np.cos(yaw*0.5)
    sy = np.sin(yaw*0.5)
    cp = np.cos(pitch*0.5)
    sp = np.sin(pitch*0.5)
    cr = np.cos(roll*0.5)
    sr = np.sin(roll*0.5)
    return [
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy
    ]

def axis_angle2quat(axis, angle):
    return pb.getQuaternionFromAxisAngle(axis, angle)

def quat2axis_angle(q):
    return pb.getAxisAngleFromQuaternion(q)

def rotate_vector(v, q):
    return list(pb.rotateVector(q, v))

def slerp(q0, q1, t):
    return list(pb.getQuaternionSlerp(q0, q1, t))

def lerp(v0, v1, t):
    if hasattr(v0, "__len__"):
        return [(1-t)*_v0 + t*_v1 for _v0, _v1 in zip(v0, v1)]
    return (1-t)*v0 + t*v1

def orient2heading(orient, up_dir):
    if up_dir == 1:
        _, _, heading = quat2euler_zyx((orient[0], -orient[2], orient[1], orient[3]))
    elif up_dir == 2:
        _, _, heading = quat2euler_zyx(orient)
    else:
        assert(up_dir in [1, 2])
    return heading
    
def so_fb_butter_lpf(samples, fs, fc):
    # second-order forward-backward butterworth low-pass filter 
    c = 1.0 / np.tan(np.pi*fc/fs)
    a0 = 1.0 / (1.0 + np.sqrt(2)*c + c*c)
    a1 = 2 * a0
    a2 = a0
    b1 = 2 * a0 * (1 - c*c)
    b2 = a0 * (1 - np.sqrt(2)*c + c*c)
    
    x1, x2, y1, y2 = samples[0], samples[0], samples[0], samples[0]
    ys = []
    for x in samples:
        y = a0*x + a1*x1 + a2*x2 - b1 * y1 - b2*y2
        x2 = x1
        x1 = x
        y2 = y1
        y1 = y
        ys.append(y)
    
    res = []
    y1, y2, x1, x2 = ys[-1], ys[-1], ys[-1], ys[-1]
    for y in reversed(ys):
        x = a0*y + a1*y1 + a2*y2 - b1 * x1 - b2*x2
        y2 = y1
        y1 = y
        x2 = x1
        x1 = x
        res.append(x)
    
    return list(reversed(res))

def spd_controller(env, obj, tar_p, tar_dp, kp, kd, dt):
    pos, orient = env.get_base_position_and_orientation(obj)
    linear_vel, angular_vel = env.get_base_velocity(obj)
    q = list(pos) + list(orient)
    dq = list(linear_vel) + list(angular_vel)
    dq.append(0)

    e_p, e_dp = [0]*7, [0]*7
    kp, kd = [0]*7 + kp, [0]*7 + kd
    i = 0

    for s in env.get_joint_states_multi_dof(obj, range(env.get_num_joints(obj))):
        pos, vel = s[0], s[1]
        if len(pos) == 1:
            q.append(pos[0])
            dq.append(vel[0])
            e_p.append(tar_p[i] - (pos[0]+vel[0]*dt))
            e_dp.append(tar_dp[i] - vel[0])
            i += 1
        elif len(pos) == 4:
            q.extend(pos)
            dq.extend(vel)
            dq.append(0)
            pos = np.add(pos, np.multiply(vel2quat_diff(vel, pos), dt))
            axis, angle = quat2axis_angle(
                quatdiff_rel(pos, tar_p[i:i+4])
            )
            e_p.extend([
                axis[0]*angle, axis[1]*angle, axis[2]*angle, 0
            ])
            e_dp.extend([
                tar_dp[i]-vel[0], tar_dp[i+1]-vel[1], tar_dp[i+2]-vel[2], 0
            ])
            i += 4
        elif len(pos) > 0:
            assert(len(pos) == 1 or len(pos) == 4)
    kp_e_p_add_kd_e_dp = np.add(np.multiply(kp, e_p), np.multiply(kd, e_dp))

    m = env.calculate_mass_matrix(obj, q, flags=1)
    m = np.array(m) + np.diag(np.multiply(kd, dt))

    c = env.calculate_inverse_dynamics(obj, q, dq, [0]*len(q), flags=1)
    f = kp_e_p_add_kd_e_dp - c

    a = np.linalg.solve(m, f)
    tau = kp_e_p_add_kd_e_dp - kd*a * dt
    return tau[7:]

