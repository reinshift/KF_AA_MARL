import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.distributions.normal import Normal

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, isorthogonalize=False):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        if isorthogonalize:
            self.orthogonal_init()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = torch.tanh(self.mean(x))
        std = torch.exp(self.log_std.expand_as(mean))
        # std = torch.clamp(std, min=0.01, max=0.5) # optional: std range [0.01, 0.5], %68 of the samples are within (mean ± std)
        return mean, std
    
    def orthogonal_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if layer == self.mean:
                    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                else:
                    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(layer.bias, 0)

class Critic(nn.Module):
    def __init__(self, obs_dim, isorthogonalize=False):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

        if isorthogonalize:
            self.orthogonal_init()

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value
    
    def orthogonal_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if layer == self.value:
                    nn.init.orthogonal_(layer.weight)
                else:
                    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(layer.bias, 0)

class MAPPOAgent:
    def __init__(self, obs_dim, action_dim, lr, gamma, lam, epsilon, device, 
                 ifadvNorm=False, if_lr_decay=False, iforthogonalize=False, total_episodes=0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.device = device
        self.ifadvNorm = ifadvNorm
        self.if_lr_decay = if_lr_decay
        self.total_episodes = total_episodes

        self.actor = Actor(obs_dim, action_dim, iforthogonalize).to(device)
        self.critic = Critic(obs_dim, iforthogonalize).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        if self.if_lr_decay and self.total_episodes > 0:
            self.scheduler_actor = torch.optim.lr_scheduler.LambdaLR(
                self.actor_optimizer, 
                lr_lambda=lambda epoch: 1 - epoch / self.total_episodes
            )
            self.scheduler_critic = torch.optim.lr_scheduler.LambdaLR(
                self.critic_optimizer, 
                lr_lambda=lambda epoch: 1 - epoch / self.total_episodes
            )
        else:
            self.scheduler_actor = None
            self.scheduler_critic = None

    def choose_action(self, obs, a_max=0.04):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        mean, std = self.actor(obs)
        # print(f"Mean: {mean}, Std: {std}")
        dist = Normal(mean, std)
        action = dist.sample()
        action_norm = torch.norm(action, p=2)
        log_prob = dist.log_prob(action).sum(-1)

        if action_norm > a_max:
            action = action * (a_max / action_norm)
        
        value = self.critic(obs).squeeze(0)
        return action.cpu().numpy().flatten(), log_prob.item(), value.item()

    def update(self, buffer):
        obs, actions, rewards, dones, log_probs_old, values, next_values = buffer.sample()
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        log_probs_old = log_probs_old.to(self.device)
        values = values.to(self.device)
        next_values = next_values.to(self.device)

        advantages = torch.zeros_like(rewards)
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t].float()) - values[t]
            advantage = delta + self.gamma * self.lam * (1.0 - dones[t].float()) * advantage
            advantages[t] = advantage

        returns = advantages + values

        for _ in range(10):  # Number of epochs
            # Sample batches
            batch_size = 256
            indices = np.random.permutation(len(rewards))
            for i in range(0, len(rewards), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]

                # trick1: advantage normalization
                if self.ifadvNorm:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # Compute new values
                new_values = self.critic(batch_obs).squeeze(-1)

                # Compute critic loss
                critic_loss = (new_values - batch_returns).pow(2).mean()

                # Compute new policy
                new_mean, new_std = self.actor(batch_obs)
                new_dist = Normal(new_mean, new_std)
                new_log_probs = new_dist.log_prob(batch_actions).sum(-1)

                # Compute ratio
                ratio = (new_log_probs - batch_log_probs_old).exp()

                # Compute surrogate loss
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # Optional
                entropy = new_dist.entropy().sum(-1).mean()
                actor_loss -= 0.01 * entropy

                # Update actor and critic
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
        
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # optional: grad clip
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5) 
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5) 

                if self.scheduler_actor is not None:
                    self.scheduler_actor.step()
                if self.scheduler_critic is not None:
                    self.scheduler_critic.step()
            
    def save_model(self, save_dir, agent_id, agent_type):
            """
            save actor & critic models to given directory

            :param save_dir: directory to save models
            :param agent_id: agent id
            :param agent_type: type('hunter' or 'target')
            """
            agent_folder = os.path.join(save_dir, f"{agent_type}_{agent_id}")
            os.makedirs(agent_folder, exist_ok=True)

            actor_path = os.path.join(agent_folder, "actor.pth")
            critic_path = os.path.join(agent_folder, "critic.pth")

            torch.save(self.actor.state_dict(), actor_path)
            torch.save(self.critic.state_dict(), critic_path)

            print(f"Saved {agent_type} {agent_id} models to {agent_folder}")

    def load_model(self, model_dir, agent_id, agent_type):
        """
        from certain folder to load actor & critic models' parameters

        :param model_dir: directory to load models
        :param agent_id: agent id
        :param agent_type: type('hunter' or 'target')
        """
        agent_folder = os.path.join(model_dir, f"{agent_type}_{agent_id}")
        actor_path = os.path.join(agent_folder, "actor.pth")
        critic_path = os.path.join(agent_folder, "critic.pth")

        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            raise FileNotFoundError(f"file not found: {actor_path} or {critic_path}")

        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

        self.actor.eval()
        self.critic.eval()

        print(f"Loaded {agent_type} {agent_id} models")