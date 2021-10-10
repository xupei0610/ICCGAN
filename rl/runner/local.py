import os, time
import torch

import numpy as np
from fastdtw import fastdtw

from typing import Callable, Any, Optional, Iterable, Sequence, Mapping
import multiprocessing
from ..agent import Agent
Env = Any

from .. import utils


def worker_name(rank: int):
    name = "Evaluator" if rank < 0 else "Worker"
    return "{}{}".format(name, abs(rank))

def runner(
    work_fn: Callable,
    env_wrapper: Callable,
    agent_wrapper: Callable[[Env], Agent],
    seed: int = None,
    args: Sequence=[], kwargs: Mapping[str, Any]=dict()
):
    env = env_wrapper()
    if seed is not None:
        utils.seed(seed, env)
    agent = agent_wrapper(env)
    agent.init()
    return work_fn(env, agent, *args, **kwargs)

def train(env: Env, agent: Agent):
    done = True
    agent.eval()
    while not agent.requests_quit:
        s = env.reset() if done else s_
        a, *args = agent.act(s, True)
        s_, r, done, info = env.step(a)
        agent.store(s, a, r, s_, done, info, *args)
        if agent.needs_update():
            agent.train()
            agent.update()
            agent.eval()

def evaluate(env: Env, agent: Agent,
    seed: Optional[int] = None,
    trials: int = 10,
    child_processes: Iterable[multiprocessing.context.Process] = None,
    timeout: int = 3600,
    terminate_signal: Optional[multiprocessing.Value] = None,
    keep_best_checkpoint: bool = True,
    backup_interval: int = 0
):
    def pose_dist(p, p_):
        return np.mean(np.linalg.norm(p - p_, axis=-1))

    agent.eval()
    done = True
    lifetime = 0.
    pose_err = []
    tries = -1
    finished, global_step = False, -1
    best_pose_err = 9999999
    if seed is not None: utils.seed(seed, env)
    while True:
        if tries < 0:
            last_response_time = time.time()
            ckpt = None
            while not finished:
                if os.path.exists(agent.checkpoint_file):
                    try:
                        ckpt = torch.load(agent.checkpoint_file, map_location=agent.device)
                    except Exception as e:
                        print("Evaluator error:", e)
                        ckpt = None
                    if ckpt:
                        agent.load_state_dict(ckpt)
                        step = agent.global_step.item()
                        if step <= global_step:
                            ckpt = None
                        else:
                            global_step = step
                            break
                finished = False
                if child_processes:
                    finished = not all(p.is_alive() for p in child_processes)
                if not finished and timeout and timeout > 0:
                    finished = time.time() - last_response_time > timeout
                if not finished and terminate_signal is not None:
                    finished = terminate_signal.value
                time.sleep(30)
            if finished: break
            tries = 0
            utils.env_seed(env, seed)

        if done:
            # env.controllable = False
            # env.random_init_pose = False
            # env.overtime = 120
            s = env.reset()
            pos_frames, pos_frames_ref = [], []
            if env.ref_motion.loopable:
                cycle_len = min(env.overtime, len(env.ref_motion.frames))
            else:
                cycle_len = min(100, len(env.ref_motion.frames))
            already_done = False
        else:
            s = s_

        a  = agent.act(s, False)[0]
        s_, _, done, _ = env.step(a)
        
        if not already_done:
            lifetime += 1
            already_done = done

        if len(pos_frames) < cycle_len:
            pos = [s[0] for s in env.agent.link_state(env.agent.motors)]
            pos_frames.append(np.array(pos))
            pos_ref = [s[0] for s in env.ref_motion.agent.link_state(env.agent.motors)]
            pos_frames_ref.append(np.array(pos_ref))
            done = False
        elif already_done:
            done = True

        if done:
            dist, _ = fastdtw(pos_frames[:cycle_len], pos_frames_ref[:cycle_len], dist=pose_dist)
            dist /= min(len(pos_frames), cycle_len)
            pose_err.append(dist)
        tries += done
        if tries >= trials:
            lifetime /= tries
            pose_err = np.mean(pose_err)
            if agent.logger:
                agent.logger.add_scalar("eval/pose_err", pose_err, global_step)
                agent.logger.add_scalar("eval/lifetime", lifetime, global_step)
                agent.logger.add_scalar("eval/samples", agent.samples.item(), global_step)
            print("[PERFORM] Step: {:.0f}; Life Time: {}; Pose Error: {:.4f}; Samples: {:.0f}; {}".format(
                agent.global_step, round(lifetime, 2), pose_err, agent.samples.item(),
                time.strftime("%m-%d %H:%M:%S")
            ))

            if backup_interval > 0:
                cache_id = int(agent.samples.item()) // int(backup_interval)
                if cache_id:
                    cache_file = agent.checkpoint_file+"-{}".format(cache_id)
                    if not os.path.exists(cache_file):
                        torch.save(ckpt, cache_file)
            if keep_best_checkpoint and pose_err < best_pose_err:
                torch.save(ckpt, agent.checkpoint_file+"-best")
                best_pose_err = pose_err

            pose_err = []
            lifetime = 0
            tries = -1