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


DEFAULT_CURRICULUM = {
    'enabled': True,
    'stages': [
        {
            'name': 'pursuit_avoidance',
            'until_fraction': 0.30,
            'reward': {
                'capture_reward': 12.0,
                'chase_reward_coeff': 0.6,
                'escape_reward_coeff': 0.2,
                'alignment_reward_coeff': 0.0,
                'safe_penalty_coeff': 0.8,
                'obstacle_interior_penalty': 1.2,
                'distance_threshold': 0.02,
            },
            'ablation': {
                'use_density_field': False,
                'use_role_assignment': False,
                'use_ref_velocity': False,
            },
        },
        {
            'name': 'assignment_enabled',
            'until_fraction': 0.55,
            'reward': {
                'capture_reward': 12.0,
                'chase_reward_coeff': 0.35,
                'escape_reward_coeff': 0.3,
                'alignment_reward_coeff': 0.0,
                'safe_penalty_coeff': 0.6,
                'obstacle_interior_penalty': 1.0,
                'distance_threshold': 0.018,
            },
            'ablation': {
                'use_density_field': True,
                'use_role_assignment': False,
                'use_ref_velocity': False,
            },
        },
        {
            'name': 'escape_guidance',
            'until_fraction': 0.80,
            'reward': {
                'capture_reward': 10.0,
                'chase_reward_coeff': 0.2,
                'escape_reward_coeff': 0.6,
                'alignment_reward_coeff': 0.1,
                'safe_penalty_coeff': 0.45,
                'obstacle_interior_penalty': 0.8,
                'distance_threshold': 0.015,
            },
            'ablation': {
                'use_density_field': True,
                'use_role_assignment': False,
                'use_ref_velocity': True,
            },
        },
        {
            'name': 'full_coordination',
            'until_fraction': 1.00,
            'reward': {
                'capture_reward': 10.0,
                'chase_reward_coeff': 0.1,
                'escape_reward_coeff': 1.0,
                'alignment_reward_coeff': 0.2,
                'safe_penalty_coeff': 0.3,
                'obstacle_interior_penalty': 0.5,
                'distance_threshold': 0.012,
            },
            'ablation': {
                'use_density_field': True,
                'use_role_assignment': True,
                'use_ref_velocity': True,
            },
        },
    ],
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _clone_curriculum(curriculum):
    if curriculum is None:
        curriculum = DEFAULT_CURRICULUM
    return {
        'enabled': curriculum.get('enabled', True),
        'stages': [dict(stage) for stage in curriculum.get('stages', [])],
    }


def resolve_curriculum_stage(curriculum, episode, total_episodes):
    if not curriculum.get('enabled', True):
        return None

    progress = episode / max(total_episodes, 1)
    stages = curriculum.get('stages', [])
    if not stages:
        return None

    for stage in stages:
        if progress <= stage.get('until_fraction', 1.0):
            return stage
    return stages[-1]


def apply_curriculum_stage(env, stage):
    if stage is None:
        return "default"
    env.configure_training_phase(
        stage_name=stage.get('name'),
        reward_config=stage.get('reward'),
        ablation_config=stage.get('ablation'),
    )
    return stage.get('name', 'default')

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

    curriculum = _clone_curriculum(getattr(args, 'curriculum', None))

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

    # initialize CSV file with extended columns
    rewards_csv_path = os.path.join(data_dir, "rewards.csv")
    with open(rewards_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "stage", "steps", "capture_success",
                         "total_reward_hunters", "total_reward_targets",
                         "avg_chase_reward", "avg_capture_reward",
                         "avg_escape_reward", "avg_alignment_reward",
                         "avg_critic_loss", "avg_actor_loss"])

    update_counter = 0
    score_threshold = args.score_threshold
    active_stage_name = None
    for episode in range(1, args.num_episodes + 1):
        stage = resolve_curriculum_stage(curriculum, episode, args.num_episodes)
        stage_name = apply_curriculum_stage(env, stage)
        if stage_name != active_stage_name:
            active_stage_name = stage_name
            print(f"Switched curriculum stage -> {active_stage_name}")

        h_obs, t_obs = env.reset()
        episode_rewards_hunters = np.zeros(env.num_hunters)
        episode_rewards_targets = np.zeros(env.num_targets)
        done = False
        current_step = 1

        # Per-episode tracking
        episode_chase = []
        episode_capture = []
        episode_escape = []
        episode_alignment = []
        episode_critic_losses = []
        episode_actor_losses = []
        capture_success = False

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
            h_next_obs, t_next_obs, rewards, dones, reward_info = env.step(actions)

            # Track reward components
            episode_chase.append(np.mean(reward_info['chase_rewards']))
            episode_capture.append(np.mean(reward_info['capture_rewards']))
            episode_escape.append(np.mean(reward_info['escape_rewards']))
            episode_alignment.append(np.mean(reward_info['alignment_rewards']))

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

            done = any(dones)
            if reward_info.get('capture_happened', False):
                capture_success = True

            update_counter += 1
            if update_counter % args.update_freq == 0:
                if hunters_buffer.size() >= args.min_buffer_size:
                    for _ in range(args.update_iterations):
                        batch = hunters_buffer.sample(args.batch_size)
                        for hunter in hunters:
                            losses = hunter.update(batch)
                            if losses is not None:
                                episode_critic_losses.append(losses[0])
                                episode_actor_losses.append(losses[1])
                if targets_buffer.size() >= args.min_buffer_size:
                    for _ in range(args.update_iterations):
                        batch = targets_buffer.sample(args.batch_size)
                        for target in targets:
                            target.update(batch)

        total_reward_hunters = episode_rewards_hunters.sum()
        total_reward_targets = episode_rewards_targets.sum()
        ep_steps = current_step - 1

        # Compute averages for logging
        avg_chase = np.mean(episode_chase) if episode_chase else 0.0
        avg_capture = np.mean(episode_capture) if episode_capture else 0.0
        avg_escape = np.mean(episode_escape) if episode_escape else 0.0
        avg_alignment = np.mean(episode_alignment) if episode_alignment else 0.0
        avg_critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0.0
        avg_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0.0

        cap_str = "CAPTURED" if capture_success else ""
        print(f"Episode {episode}/{args.num_episodes}, "
              f"Stage: {active_stage_name}, "
              f"Steps: {ep_steps}, "
              f"H_Reward: {total_reward_hunters:.2f}, "
              f"T_Reward: {total_reward_targets:.2f} "
              f"{cap_str}")

        with open(rewards_csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([episode, active_stage_name, ep_steps, int(capture_success),
                             f"{total_reward_hunters:.4f}", f"{total_reward_targets:.4f}",
                             f"{avg_chase:.4f}", f"{avg_capture:.4f}",
                             f"{avg_escape:.4f}", f"{avg_alignment:.4f}",
                             f"{avg_critic_loss:.6f}", f"{avg_actor_loss:.6f}"])

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
    parser.add_argument('--t_actor_dim', type=int, default=36, help='dimension of targets\' observation')
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
