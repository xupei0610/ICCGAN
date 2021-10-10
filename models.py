import torch
import numpy as np

from rl.policy import GaussianPolicy
from rl.ppo import PPO
from rl import utils
from rl.buffer import OnPolicyBuffer


class Critic(torch.nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        self.state_shape = state_shape
        hidden_dim = 256
        self.rnn = torch.nn.GRU(self.state_shape[-1], hidden_dim, 1, batch_first=True)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1024),
            torch.nn.ReLU6(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU6(inplace=True),
            torch.nn.Linear(512, 1),
        )
        for n, p in self.model.named_parameters():
            if "bias" in n:
                torch.nn.init.constant_(p, 0.)
            elif "weight" in n:
                p.data.copy_(torch.fmod(torch.randn(p.shape),2)*0.01)
        self.seq_len : int or torch.tensor = 0
        
    def forward(self, s):
        if s.ndim < 3: s = s.view(1, *s.shape)
        out, _ = self.rnn(s)
        if torch.is_tensor(self.seq_len):
            with torch.no_grad():
                seq_len = self.seq_len-1
                seq_len.unsqueeze_(1)
                seq_len.unsqueeze_(-1)
                seq_len = seq_len.expand(-1, -1, *out.shape[2:])
            x = out.gather(1, seq_len).squeeze(1)
        else:
            x = out[:, self.seq_len-1]
        return self.model(x)


class Actor(torch.nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.state_shape = state_shape
        hidden_dim = 256
        self.rnn = torch.nn.GRU(self.state_shape[-1], hidden_dim, 1, batch_first=True)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1024),
            torch.nn.ReLU6(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU6(inplace=True),
            GaussianPolicy(512, action_shape[0], init_std=0.05)
        )
        for n, p in self.model.named_parameters():
            if "std" not in n:
                if "bias" in n:
                    torch.nn.init.constant_(p, 0.)
                elif "weight" in n:
                    p.data.copy_(torch.fmod(torch.randn(p.shape),2)*0.01)
        self.seq_len : int or torch.tensor = 0

    def forward(self, s):
        if s.ndim < 3: s = s.view(1, *s.shape)
        out, _ = self.rnn(s)
        if torch.is_tensor(self.seq_len):
            with torch.no_grad():
                seq_len = self.seq_len-1
                seq_len.unsqueeze_(1)
                seq_len.unsqueeze_(-1)
                seq_len = seq_len.expand(-1, -1, *out.shape[2:])
            x = out.gather(1, seq_len).squeeze(1)
        else:
            x = out[:, self.seq_len-1]
        return self.model(x)


class Discriminator(torch.nn.Module):
    def __init__(self, ob_shape):
        super().__init__()
        self.ob_shape = ob_shape
        hidden_dim = 256
        self.rnn = torch.nn.GRU(self.ob_shape[-1], hidden_dim, 1, batch_first=True)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 32)
        )
        i = 0
        for n, p in self.model.named_parameters():
            if "bias" in n:
                torch.nn.init.constant_(p, 0.)
            elif "weight" in n:
                gain = 1 if i == 2 else 2**0.5 
                torch.nn.init.orthogonal_(p, gain=gain)
                i += 1

    def forward(self, s, seq_len: int or torch.tensor=0):
        if s.ndim < 3: s = s.view(1, *s.shape)
        out, _ = self.rnn(s)
        if torch.is_tensor(seq_len):
            with torch.no_grad():
                seq_len = seq_len-1
                seq_len.unsqueeze_(1)
                seq_len.unsqueeze_(-1)
                seq_len = seq_len.expand(-1, -1, *out.shape[2:])
            x = out.gather(1, seq_len).squeeze(1)
        else:
            x = out[:, seq_len-1]
        return self.model(x)


class ICCGAN(PPO):
    def __init__(self, discriminator_learning_rate, discriminator_network, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.buffer_D = OnPolicyBuffer([
            "agent_state", "expert_state", "seq_len"
        ], capacity=self.horizon, batch_size=self.batch_size)
        self.buffer.add("seq_len")

        self.model.discriminator = discriminator_network

        self.optimizers["discriminator"] = torch.optim.Adam(
            self.discriminator.parameters(), lr=discriminator_learning_rate
        )
        
        self.model.discriminator.state_normalizer = utils.Normalizer([self.model.discriminator.ob_shape[-1]], clamp=None)
        self.optimizers["discriminator_state_normalizer"] = utils.MovingAverage(self.model.discriminator.state_normalizer.parameters())
        self.hooks.append(utils.MovingAverageNormalizerHook(
            self.model.discriminator.state_normalizer, self.optimizers["discriminator_state_normalizer"],
            lambda : self.buffer_D["agent_state"]
        ))

    @property
    def discriminator(self):
        return self.model.discriminator

    @staticmethod
    def zero_pad(s, n):
        if len(s) < n:
            s = np.concatenate((
                s,
                np.zeros((n-s.shape[0], s.shape[1]),dtype=np.float32)
            ))
        return s
        
    def act(self, s, stochastic=None):
        return super().act(s["state"], stochastic)

    def store(self, s, a, r, s_, done, info, log_prob, v):
        a_s = s_["discriminator"]
        e_s = info["expert_state"]["discriminator"]
        s = s["state"]
        s_ = s_["state"]
        
        seq_len_ = len(a_s)
        a_s = self.zero_pad(a_s, self.discriminator.ob_shape[0])
        e_s = self.zero_pad(e_s, self.discriminator.ob_shape[0])
        self.buffer_D.store(
            agent_state=a_s, expert_state=e_s, seq_len=seq_len_
        )

        seq_len = len(s)
        s = self.zero_pad(s, self.actor.state_shape[0])
        if not done:  # normal
            boostrap_state = True
        elif utils.env_overtime(info): # overtime
            boostrap_state = s_
        else:
            boostrap_state = None   # terminate

        self.buffer.store(
            state=s, action=a, value=v, advantage=boostrap_state, log_prob=log_prob,
            seq_len=seq_len
        )
        self._needs_update = len(self.buffer) >= self.horizon if self.horizon else done
        if self._needs_update:
            if not done: self.buffer["advantage"][-1] = s_
            self.compute_reward()
    
    def update_discriminator(self):
        r_loss, f_loss = [], []
        for data in self.buffer_D:
            with torch.no_grad():
                seq_len = self.placeholder(data["seq_len"])
                fake = self.placeholder(data["agent_state"])
                real = self.placeholder(data["expert_state"])
                fake = self.model.discriminator.state_normalizer(fake)
                real = self.model.discriminator.state_normalizer(real)

            dreal = self.discriminator(real, seq_len)
            real_loss = torch.nn.functional.relu(1-dreal).mean()
            dfake = self.discriminator(fake, seq_len)
            fake_loss = torch.nn.functional.relu(1+dfake).mean()
            r_loss.append(dreal.mean().item())
            f_loss.append(dfake.mean().item())

            alpha = torch.rand(real.size(0), dtype=real.dtype, device=real.device)
            alpha = alpha.view(-1, *([1]*(real.ndim-1)))
            interpolated = alpha*real+(1-alpha)*fake
            interpolated.requires_grad = True
            with torch.backends.cudnn.flags(enabled=False):
                dout = self.discriminator(interpolated, seq_len)
            grad = torch.autograd.grad(
                dout, interpolated, torch.ones_like(dout),
                retain_graph=True, create_graph=True, only_inputs=True
            )[0]
            gp = grad.reshape(grad.size(0), -1).norm(2, dim=1).sub(1).square().mean()

            d_loss = real_loss + fake_loss + 10*gp
            if self.optimizers["discriminator"]:
                self.optimizers["discriminator"].zero_grad()
                d_loss.backward()
                self.step_opt(self.optimizers["discriminator"])

        if self.logger:
            global_step = self.global_step.item()
            self.logger.add_scalar("discriminator/score_real", sum(r_loss)/len(r_loss), global_step)
            self.logger.add_scalar("discriminator/score_fake", sum(f_loss)/len(f_loss), global_step)

    def compute_reward(self):
        self.eval()
        with torch.no_grad():
            seq_len = self.placeholder(self.buffer_D["seq_len"], dtype=torch.int64)
            fake = self.placeholder(self.buffer_D["agent_state"])
            fake = self.discriminator.state_normalizer(fake)
            rewards = self.discriminator(fake, seq_len).clamp_(-1, 1).mean(-1).cpu().numpy()
        n_episodes = 0
        self._buffer_path_ptr = 0
        for i, r in enumerate(rewards):
            s_ = self.buffer["advantage"][i]
            if s_ is None:    # terminate
                self.buffer["advantage"][i] = -1
                rewards[i] = -1
            else:
                self.buffer["advantage"][i] = r
            if s_ is not True: # boostrap
                self.update_path_advantage(s_, i+1)
                n_episodes += 1
        self.train()

        if self.logger:
            global_step = self.global_step.item()
            rewards_sum = rewards.sum()
            self.logger.add_scalar("discriminator/reward_average", rewards_sum/rewards.shape[0], global_step)
            self.logger.add_scalar("discriminator/reward_total", rewards_sum/max(1, n_episodes), global_step)

    def loss(self, data):
        with torch.no_grad():
            seq_len = self.placeholder(data["seq_len"])
            self.actor.seq_len = seq_len
            self.critic.seq_len = seq_len
        l = super().loss(data)
        self.actor.seq_len = 0
        self.critic.seq_len = 0
        return l
    
    def _update(self):
        self.update_discriminator()
        self.buffer_D.clear()
        return super()._update()
