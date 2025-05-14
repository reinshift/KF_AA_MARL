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
                 iforthogonalize=False, noise_clip=0.5, a_max=0.04, if_lr_decay=False, total_episodes=500):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.device = device
        self.noise_clip = noise_clip
        self.a_max = a_max

        self.actor = Actor(obs_dim, action_dim, iforthogonalize, a_max).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(obs_dim, action_dim, iforthogonalize).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.noise = Normal(0, noise_std)

        self.if_lr_decay = if_lr_decay
        self.total_episodes = total_episodes

    def select_action(self, obs, noise=True):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.actor(obs).detach().cpu().numpy().flatten()
        if noise:
            noise_sample = self.noise.sample(action.shape).numpy()
            # print("action: ", action) # optional: print the action for debugging
            action = action + noise_sample
            # print("noised action: ", action) # optional: print the action for debugging

        action_norm = np.linalg.norm(action, ord=2)
        if action_norm > self.a_max:
            action = action * (self.a_max / action_norm)
        return action

    def update(self, batch):
        obs, actions, rewards, next_obs, dones = batch

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
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(obs, self.actor(obs))[0].mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # lr decay
        if self.if_lr_decay and self.total_episodes > 0:
            self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.actor_optimizer,
                lr_lambda=lambda epoch: 1 - epoch / self.total_episodes
            )
            self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.critic_optimizer,
                lr_lambda=lambda epoch: 1 - epoch / self.total_episodes
            )
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None

        # Update target networks
        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic, self.critic_target, self.tau)

    def soft_update(self, source, target, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

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