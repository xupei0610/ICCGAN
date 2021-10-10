import os
from functools import partial

import torch

from rl.runner.local import runner, evaluate, train
from rl import utils

from envs import MotionImitate, Observation
from models import Actor, Critic, Discriminator, ICCGAN

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("act", nargs='+')
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--max_samples", type=float, default=int(2e7))
parser.add_argument("--world_size", type=int, default=8)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--rank", type=int, default=None)
parser.add_argument("--master_addr", type=str, default="127.0.0.1")
parser.add_argument("--master_port", type=str, default="29501")
settings = parser.parse_args()

WORLD_SIZE = settings.world_size
CHIEF_RANK = 0
CHECKPOINT_DIR = "ckpt_{}".format("_".join(settings.act) if settings.ckpt is None else settings.ckpt)
CHECKPOINT_FILE = "{}/ckpt".format(CHECKPOINT_DIR)
LOG_DIR = None if settings.test else CHECKPOINT_DIR
HORIZON = 4096//WORLD_SIZE
BATCH_SIZE = 256//WORLD_SIZE

OBSERVATION_HISTORY = 4
DISCRIMINATOR_OBSERVATION_HISTORY = OBSERVATION_HISTORY + 1

def env_wrapper(**kwargs):
    return MotionImitate(action=settings.act, observation={
        "discriminator": Observation(DISCRIMINATOR_OBSERVATION_HISTORY, with_velocity=False),
        "state": Observation(OBSERVATION_HISTORY, with_velocity=True)
    }, **kwargs)

def agent_wrapper(env, rank, device=None, hooks=[]):
    is_chief = rank == CHIEF_RANK
    is_evaluator = rank < 0
    name = "evaluator{}".format(abs(rank)) if is_evaluator else "worker{}".format(rank)
    log_dir = os.path.join(LOG_DIR, name) if (is_chief or is_evaluator) and LOG_DIR else None

    state_shape = [OBSERVATION_HISTORY, env.observation_space["state"].shape[-1]]
    ob_shape = [DISCRIMINATOR_OBSERVATION_HISTORY, env.observation_space["discriminator"].shape[-1]]

    return ICCGAN(
        discriminator_learning_rate=1e-5,
        critic_learning_rate=1e-4,
        actor_learning_rate=5e-6,
        discriminator_network=Discriminator(ob_shape),
        critic_network=Critic(state_shape),
        actor_network=Actor(state_shape, env.action_space.shape),

        value_loss_coef=0.5,
        gamma=0.95,
        clip_grad_norm=1,
        normalize_state=[state_shape[-1]],
        clip_state=5.,
        opt_epoch=5,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
        checkpoint_file=CHECKPOINT_FILE,
        log_dir=log_dir,
        checkpoint_save_interval=1000,
        max_samples=int(settings.max_samples),
        is_chief=is_chief,
        hooks=[h() for h in hooks],
        device=device
    )

def run_evaluator(render, child_processes=None):
    runner(evaluate, partial(env_wrapper, render=render), partial(agent_wrapper, rank=-1, device=settings.device), 
        kwargs=dict(seed=settings.seed+WORLD_SIZE, child_processes=child_processes))


if __name__ == "__main__":
    if settings.test:
        run_evaluator(render=True)
    else:
        utils.set_cuda_device(settings.device)
        from rl.runner.distributed import distributed, DistributedSyncHook
        args = (partial(runner, train, env_wrapper,
                    partial(agent_wrapper, rank=settings.rank, device=settings.device, hooks=[DistributedSyncHook]), settings.seed+settings.rank
                ), "gloo", settings.rank, WORLD_SIZE)
        kwargs = dict(
                default_cuda_device=settings.device, 
                master_addr=settings.master_addr, master_port=settings.master_port
            )
        if settings.rank == 0:
            processes = []
            torch.multiprocessing.set_start_method("spawn", force=True)
            p = torch.multiprocessing.Process(target=distributed, args=args, kwargs=kwargs)
            p.start()
            processes.append(p)
            run_evaluator(render=False, child_processes=processes)
        else:
            distributed(*args, **kwargs)
