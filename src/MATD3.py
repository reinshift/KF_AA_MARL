import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.distributions.normal import Normal
import copy

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, isorthogonalize=False, a_max=0.01):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

        if isorthogonalize:
            self.orthogonal_init()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def orthogonal_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(layer.bias, 0)

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, isorthogonalize=False):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.q1 = nn.Linear(128, 1)

        self.fc3 = nn.Linear(obs_dim + action_dim, 128)
        self.fc4 = nn.Linear(128, 128)
        self.q2 = nn.Linear(128, 1)

        if isorthogonalize:
            self.orthogonal_init()

    def forward(self, obs, actions):
        x1 = torch.cat([obs, actions], dim=-1)
        x1 = torch.relu(self.fc1(x1))
        x1 = torch.relu(self.fc2(x1))
        q1 = self.q1(x1)

        x2 = torch.cat([obs, actions], dim=-1)
        x2 = torch.relu(self.fc3(x2))
        x2 = torch.relu(self.fc4(x2))
        q2 = self.q2(x2)
        return q1, q2

    def orthogonal_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

class MATD3Agent:
    def __init__(self, obs_dim, action_dim, lr, gamma, tau, noise_std, device,
                 iforthogonalize=False, noise_clip=0.5, a_max=0.04, if_lr_decay=False, total_episodes=500,
                 policy_delay=2, grad_clip_norm=0.5):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.device = device
        self.noise_clip = noise_clip
        self.a_max = a_max
        self.policy_delay = policy_delay
        self.grad_clip_norm = grad_clip_norm
        self._update_step = 0  # 用于延迟策略更新计数

        self.actor = Actor(obs_dim, action_dim, iforthogonalize, a_max).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(obs_dim, action_dim, iforthogonalize).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.noise = Normal(0, noise_std)

        # Reward normalization (running mean/std)
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0

        # lr decay
        self.if_lr_decay = if_lr_decay
        self.total_episodes = total_episodes
        if if_lr_decay and total_episodes > 0:
            self.actor_scheduler = optim.lr_scheduler.LambdaLR(
                self.actor_optimizer,
                lr_lambda=lambda step: max(0.1, 1 - step / (total_episodes * 100))
            )
            self.critic_scheduler = optim.lr_scheduler.LambdaLR(
                self.critic_optimizer,
                lr_lambda=lambda step: max(0.1, 1 - step / (total_episodes * 100))
            )
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None

    def select_action(self, obs, noise=True):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            action = self.actor(obs).squeeze(0).cpu().numpy()
        if noise:
            noise_sample = self.noise.sample(action.shape).numpy()
            # print("action: ", action) # optional: print the action for debugging
            action = action + noise_sample
            # print("noised action: ", action) # optional: print the action for debugging

        action_norm = np.linalg.norm(action, ord=2)
        if action_norm > self.a_max:
            action = action * (self.a_max / action_norm)
        return action

    def normalize_reward(self, rewards_np):
        """Running mean/std reward normalization"""
        batch_mean = rewards_np.mean()
        batch_var = rewards_np.var()
        batch_count = len(rewards_np)

        # Welford online update
        delta = batch_mean - self._reward_mean
        total_count = self._reward_count + batch_count
        if total_count == 0:
            return rewards_np
        self._reward_mean += delta * batch_count / total_count
        m_a = self._reward_var * self._reward_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self._reward_count * batch_count / total_count
        self._reward_var = m2 / total_count
        self._reward_count = total_count

        std = max(self._reward_var ** 0.5, 1e-6)
        return (rewards_np - self._reward_mean) / std

    def update(self, batch):
        obs, actions, rewards, next_obs, dones = batch

        # Reward normalization
        rewards = self.normalize_reward(rewards)

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute target Q values
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            noise = torch.clamp(self.noise.sample(next_actions.shape), -self.noise_clip, self.noise_clip).to(self.device)
            next_actions = next_actions + noise

            next_action_norm = torch.norm(next_actions, p=2, dim=-1, keepdim=True)
            next_actions = torch.where(next_action_norm > self.a_max, next_actions * (self.a_max / next_action_norm), next_actions)

            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target)
            target_q = rewards + (1 - dones) * self.gamma * q_target

        # Update critic
        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = (current_q1 - target_q).pow(2).mean() + (current_q2 - target_q).pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        # Delayed policy update (standard TD3: update actor every policy_delay steps)
        self._update_step += 1
        actor_loss_val = None
        if self._update_step % self.policy_delay == 0:
            actor_loss = -self.critic(obs, self.actor(obs))[0].mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            # Update target networks (only when actor updates)
            self.soft_update(self.actor, self.actor_target, self.tau)
            self.soft_update(self.critic, self.critic_target, self.tau)

        # lr decay step
        if self.critic_scheduler:
            self.critic_scheduler.step()
        if self.actor_scheduler and actor_loss_val is not None:
            self.actor_scheduler.step()

        return critic_loss.item(), actor_loss_val

    def soft_update(self, source, target, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save_checkpoint(self):
        """导出完整训练状态（含optimizer和target网络）"""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }

    def load_checkpoint(self, state):
        """恢复完整训练状态"""
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])

    def save_model(self, save_dir, agent_id, agent_type):
        agent_folder = os.path.join(save_dir, f"{agent_type}_{agent_id}")
        os.makedirs(agent_folder, exist_ok=True)

        actor_path = os.path.join(agent_folder, "actor.pth")
        critic_path = os.path.join(agent_folder, "critic.pth")

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

        print(f"Saved {agent_type} {agent_id} models to {agent_folder}")

    def load_model(self, model_dir, agent_id, agent_type):

        model_dir = os.path.normpath(model_dir)
        agent_folder = os.path.join(model_dir, f"{agent_type}_{agent_id}")
        actor_path = os.path.join(agent_folder, "actor.pth")
        critic_path = os.path.join(agent_folder, "critic.pth")

        print(f"trying to load models from:\n{actor_path}\n{critic_path}")
        
        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            raise FileNotFoundError(f"file not found: {actor_path} or {critic_path}")

        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

        print(f"Loaded {agent_type} {agent_id} models")


def save_training_checkpoint(path, episode, hunters, targets, update_counter, best_score):
    """保存完整训练状态到checkpoint文件，支持断点续训"""
    torch.save({
        'episode': episode,
        'update_counter': update_counter,
        'best_score': best_score,
        'hunters': [h.save_checkpoint() for h in hunters],
        'targets': [t.save_checkpoint() for t in targets],
    }, path)


def load_training_checkpoint(path, hunters, targets):
    """从checkpoint恢复训练状态，返回 (episode, update_counter, best_score)"""
    ckpt = torch.load(path, map_location='cpu')
    for i, h in enumerate(hunters):
        h.load_checkpoint(ckpt['hunters'][i])
    for i, t in enumerate(targets):
        t.load_checkpoint(ckpt['targets'][i])
    return ckpt['episode'], ckpt['update_counter'], ckpt['best_score']
