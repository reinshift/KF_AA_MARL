import argparse
import numpy as np
import torch
from MultiTargetEnv import MultiTarEnv, set_global_seeds
from MATD3 import MATD3Agent
from replaybuffer import ReplayBuffer
import matplotlib.pyplot as plt
import os
import csv
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    set_global_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize environment
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
                          a_max=args.a_max,
                          if_lr_decay=args.iflrdecay,
                          total_episodes=args.num_episodes) for _ in range(env.num_hunters)]

    targets = [MATD3Agent(obs_dim=args.t_actor_dim,
                          action_dim=args.action_dim,
                          lr=args.lr,
                          gamma=args.gamma,
                          tau=args.tau,
                          noise_std=args.noise_std,
                          device=device,
                          iforthogonalize=args.iforthogonalize,
                          noise_clip=args.noise_clip,
                          a_max=args.a_max,
                          if_lr_decay=args.iflrdecay,
                          total_episodes=args.num_episodes) for _ in range(env.num_targets)]

    # Load models from checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading models from checkpoint: {args.checkpoint}")
        try:
            for i, hunter in enumerate(hunters):
                hunter.load_model(args.checkpoint, agent_id=i, agent_type='hunter')
            for i, target in enumerate(targets):
                target.load_model(args.checkpoint, agent_id=i, agent_type='target')
            print("Successfully loaded models from checkpoint")
        except Exception as e:
            print(f"Error loading models from checkpoint: {e}")
            print("Training will start with newly initialized models")

    # Initialize replay buffers for hunters & targets
    hunters_buffer = ReplayBuffer(max_size=args.buffer_size,
                                 obs_dim=args.h_actor_dim,
                                 action_dim=args.action_dim)

    targets_buffer = ReplayBuffer(max_size=args.buffer_size,
                                 obs_dim=args.t_actor_dim,
                                 action_dim=args.action_dim)

    # create folder for saving history rewards
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = os.path.join(os.getcwd(), "data_train", timestamp)
    os.makedirs(data_dir, exist_ok=True)

    # initialize CSV file
    rewards_csv_path = os.path.join(data_dir, "rewards.csv")
    with open(rewards_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "total_reward_hunters", "total_reward_targets"])

    update_counter = 0
    score_threshold = args.score_threshold
    for episode in range(1, args.num_episodes + 1):
        h_obs, t_obs = env.reset()
        episode_rewards_hunters = np.zeros(env.num_hunters)
        episode_rewards_targets = np.zeros(env.num_targets)
        done = False
        current_step = 1

        while (not done) and (current_step <= args.max_steps):
            actions_hunters = []
            actions_targets = []

            # hunters choose action
            for i, hunter in enumerate(hunters):
                action = hunter.select_action(h_obs[i])
                actions_hunters.append(action)

            # targets choose action
            for i, target in enumerate(targets):
                action = target.select_action(t_obs[i])
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

            # store transitions in Buffer
            for i in range(env.num_hunters):
                hunters_buffer.store_transition(h_obs[i], actions_hunters[i], rewards_hunters[i], h_next_obs[i], dones_hunters[i])

            for i in range(env.num_targets):
                targets_buffer.store_transition(t_obs[i], actions_targets[i], rewards_targets[i], t_next_obs[i], dones_targets[i])

            episode_rewards_hunters += rewards_hunters
            episode_rewards_targets += rewards_targets

            h_obs = h_next_obs
            t_obs = t_next_obs

            done = all(dones)

            update_counter += 1
            if update_counter % args.update_freq == 0:
                if hunters_buffer.size() >= args.min_buffer_size:
                    for _ in range(args.update_iterations):
                        batch = hunters_buffer.sample(args.batch_size)
                        for hunter in hunters:
                            hunter.update(batch)
                if targets_buffer.size() >= args.min_buffer_size:
                    for _ in range(args.update_iterations):
                        batch = targets_buffer.sample(args.batch_size)
                        for target in targets:
                            target.update(batch)

        total_reward_hunters = episode_rewards_hunters.sum()
        total_reward_targets = episode_rewards_targets.sum()
        print(f"Episode {episode}/{args.num_episodes}, "
              f"Total Reward Hunters: {total_reward_hunters:.2f}, "
              f"Total Reward Targets: {total_reward_targets:.2f}")

        with open(rewards_csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([episode, total_reward_hunters, total_reward_targets])

        # save model
        should_save = False
        save_reason = ""
        if episode % args.save_frequency == 0 and episode > 0:
            should_save = True
            save_reason = f"frequency_{args.save_frequency}"
        if total_reward_hunters > score_threshold:
            score_threshold = total_reward_hunters
            should_save = True
            save_reason = f"score_{total_reward_hunters:.0f}"

        if should_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(os.getcwd(), "model",
                                    f"{timestamp}_{save_reason}")
            os.makedirs(save_dir, exist_ok=True)

            for i, hunter in enumerate(hunters):
                hunter.save_model(save_dir, agent_id=i, agent_type='hunter')

            for i, target in enumerate(targets):
                target.save_model(save_dir, agent_id=i, agent_type='target')

            print(f"Models saved at episode {episode} in {save_dir}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # env relevant
    parser.add_argument('--env_length', type=float, default=2.0, help='length of boundary (km)')
    parser.add_argument('--num_obstacle', type=int, default=5, help='number of obstacles')
    parser.add_argument('--num_hunters', type=int, default=6, help='number of hunters(>=3)')
    parser.add_argument('--num_targets', type=int, default=2, help='number of targets(>=1)')
    parser.add_argument('--h_actor_dim', type=int, default=32, help='dimension of hunters\' observation')
    parser.add_argument('--t_actor_dim', type=int, default=31, help='dimension of targets\' observation')
    parser.add_argument('--action_dim', type=int, default=2, help='action dimension')
    parser.add_argument('--a_max', type=float, default=0.01, help='maximum action value (km/s^-2)')
    parser.add_argument('--ifrender', type=str2bool, default=False, help='whether to render the environment')
    parser.add_argument('--visualizelaser', type=str2bool, default=False, help='whether to visualize laser')

    # train relevant
    parser.add_argument('--seed', type=int, default=10, help='global seed')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--tau', type=float, default=0.01, help='target network update rate')
    parser.add_argument('--noise_std', type=float, default=0.005, help='std of exploration noise')
    parser.add_argument('--buffer_size', type=int, default=10000, help='capacity of buffer')
    parser.add_argument('--min_buffer_size', type=int, default=1024, help='buffer minimum capacity before updating')
    parser.add_argument('--num_episodes', type=int, default=500, help='number of episodes')
    parser.add_argument('--max_steps', type=int, default=150, help='maximum steps per episode')
    parser.add_argument('--iforthogonalize', type=str2bool, default=True, help='whether to orthogonalize the weights')
    parser.add_argument('--iflrdecay', type=str2bool, default=False, help='whether to decay the learning rate')

    # TD3 specific
    parser.add_argument('--noise_clip', type=float, default=0.01, help='range to clip target policy noise')
    parser.add_argument('--update_freq', type=int, default=10, help='frequency of updating the network')
    parser.add_argument('--update_iterations', type=int, default=20, help='number of iterations to update per time')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')

    # save and load
    parser.add_argument('--save_frequency', type=int, default=250, help='save model every save_frequency episodes')
    parser.add_argument('--score_threshold', type=float, default=500, help='threshold to save model')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint directory for continue training')

    args = parser.parse_args()
    main(args)