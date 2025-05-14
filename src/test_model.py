import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from datetime import datetime
from MultiTargetEnv import MultiTarEnv, set_global_seeds
from MATD3 import MATD3Agent
# TODO: random seed to test generalizability 
# (problem: initial position of hunters and targets may be in obstacle)
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def test_agents(args):
    set_global_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device using: {device}")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initialize environment
    env = MultiTarEnv(length=args.env_length,
                      num_obstacle=args.num_obstacle,
                      num_hunters=args.num_hunters,
                      num_targets=args.num_targets,
                      h_actor_dim=args.h_actor_dim,
                      t_actor_dim=args.t_actor_dim,
                      action_dim=args.action_dim,
                      visualize_lasers=args.visualizelaser)

    # Initialize agents for hunters and targets
    hunters = [MATD3Agent(obs_dim=args.h_actor_dim,
                          action_dim=args.action_dim,
                          lr=args.lr,
                          gamma=args.gamma,
                          tau=args.tau,
                          noise_std=args.noise_std,
                          device=device,
                          iforthogonalize=args.iforthogonalize,
                          noise_clip=args.noise_clip,
                          a_max=args.a_max) for _ in range(env.num_hunters)]

    targets = [MATD3Agent(obs_dim=args.t_actor_dim,
                          action_dim=args.action_dim,
                          lr=args.lr,
                          gamma=args.gamma,
                          tau=args.tau,
                          noise_std=args.noise_std,
                          device=device,
                          iforthogonalize=args.iforthogonalize,
                          noise_clip=args.noise_clip,
                          a_max=args.a_max) for _ in range(env.num_targets)]

    # Load pre-trained models
    model_dir = args.model_dir 
    for i, hunter in enumerate(hunters):
        hunter.load_model(model_dir, agent_id=i, agent_type='hunter')
    for i, target in enumerate(targets):
        target.load_model(model_dir, agent_id=i, agent_type='target')

    # Test loop
    num_test_episodes = args.num_test_episodes
    for episode in range(1, num_test_episodes + 1):
        h_obs, t_obs = env.reset()
        episode_rewards_hunters = np.zeros(env.num_hunters)
        episode_rewards_targets = np.zeros(env.num_targets)
        done = False
        current_step = 1

        while (not done) and (current_step <= args.max_steps):
            actions_hunters = []
            actions_targets = []

            # hunters choose action without noise
            for i, hunter in enumerate(hunters):
                action = hunter.select_action(h_obs[i], noise=False)
                actions_hunters.append(action)

            # targets choose action without noise
            for i, target in enumerate(targets):
                action = target.select_action(t_obs[i], noise=False)
                actions_targets.append(action)

            # concatenate all actions
            actions = actions_hunters + actions_targets

            # execute all actions & interact with env
            h_next_obs, t_next_obs, rewards, dones = env.step(actions)

            if args.ifrender:
                env.render()

            current_step += 1

            rewards_hunters = rewards[:env.num_hunters]
            rewards_targets = rewards[env.num_hunters:]
            dones_hunters = dones[:env.num_hunters]
            dones_targets = dones[env.num_hunters:]

            episode_rewards_hunters += rewards_hunters
            episode_rewards_targets += rewards_targets

            h_obs = h_next_obs
            t_obs = t_next_obs

            done = all(dones)

        total_reward_hunters = episode_rewards_hunters.sum()
        total_reward_targets = episode_rewards_targets.sum()
        print(f"Test Episode {episode}/{num_test_episodes}, "
              f"Total Reward Hunters: {total_reward_hunters:.2f}, "
              f"Total Reward Targets: {total_reward_targets:.2f}")
        
        if not args.ifrender:

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Generating and saving trajectory image...")
            trajectory_path = os.path.join(output_dir, f"trajectory_{episode}_{timestamp}.png")
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot([0, env.length, env.length, 0, 0], 
                    [0, 0, env.length, env.length, 0], 
                    [0, 0, 0, 0, 0], 
                    color='black', linewidth=2)
            
            for obstacle in env.obstacles:
                cx, cy, cz, r, h = obstacle._return_obs_info()
                env._create_cylinders(ax, cx, cy, cz, r, h)
            
            for i, hunter in enumerate(env.hunters):
                if hunter.history_pos:
                    trajectory = np.array(hunter.history_pos)
                    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                            color='red', alpha=0.6, label=f'Hunter {i}')
                    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                              color='blue', marker='o', s=100, label='Start' if i == 0 else "")
                    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                              color='red', marker='x', s=100, label='End' if i == 0 else "")
            
            for i, target in enumerate(env.targets):
                if target.history_pos:
                    trajectory = np.array(target.history_pos)
                    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                            color='green', alpha=0.6, label=f'Target {i}')
                    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                              color='blue', marker='o', s=100)
                    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                              color='green', marker='x', s=100)
            
            ax.set_xlim(0, env.length)
            ax.set_ylim(0, env.length)
            ax.set_zlim(0, env.length/4)
            ax.set_title(f"Episode {episode} Trajectory")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=90, azim=0)
            ax.legend(loc='upper right')
            
            plt.savefig(trajectory_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Trajectory image saved to: {trajectory_path}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # env relevant
    parser.add_argument('--env_length', type=float, default=2.0, help='length of boundary')
    parser.add_argument('--num_obstacle', type=int, default=5, help='number of obstacles')
    parser.add_argument('--num_hunters', type=int, default=6, help='number of hunters(>=3)')
    parser.add_argument('--num_targets', type=int, default=2, help='number of targets(>=1)')
    parser.add_argument('--h_actor_dim', type=int, default=32, help='dimension of hunters\' observation')
    parser.add_argument('--t_actor_dim', type=int, default=31, help='dimension of targets\' observation')
    parser.add_argument('--action_dim', type=int, default=2, help='action dimension')
    parser.add_argument('--a_max', type=float, default=0.04, help='maximum action value')
    parser.add_argument('--ifrender', type=str2bool, default=True, help='whether to render the environment')
    parser.add_argument('--visualizelaser', type=str2bool, default=False, help='whether to visualize laser')

    # test relevant
    parser.add_argument('--seed', type=int, default=20, help='global seed')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='target network update rate')
    parser.add_argument('--noise_std', type=float, default=0.1, help='std of exploration noise')
    parser.add_argument('--max_steps', type=int, default=200, help='maximum steps per episode')
    parser.add_argument('--iforthogonalize', type=str2bool, default=True, help='whether to orthogonalize the weights')
    parser.add_argument('--num_test_episodes', type=int, default=5, help='number of test episodes')

    # TD3 specific
    parser.add_argument('--noise_clip', type=float, default=0.5, help='range to clip target policy noise')

    # save and load
    parser.add_argument('--model_dir', type=str, default='model/20241223_134708_score_544', help='directory to load pre-trained models')

    args = parser.parse_args()
    test_agents(args)