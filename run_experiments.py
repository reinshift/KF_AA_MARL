"""
论文实验运行脚本

用法:
    python run_experiments.py --experiment optimized    # 运行单个实验
    python run_experiments.py --all                     # 运行全部实验
    python run_experiments.py --plots                   # 仅生成图表
"""

import os
import sys
import argparse
import shutil
from datetime import datetime
from tqdm import tqdm

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_experiments')

# ============================================================================
# 实验配置
# ============================================================================

EXPERIMENT_CONFIGS = {
    'baseline': {
        'description': '旧奖励参数 baseline',
        'num_episodes': 50,
        'seed': 42,
        'reward': {
            'capture_reward': 2.0,
            'chase_reward_coeff': 0.8,
            'escape_reward_coeff': 0.4,
            'alignment_reward_coeff': 0.5,
            'safe_penalty_coeff': 0.7,
            'obstacle_interior_penalty': 1.0,
        },
        'ablation': {
            'use_density_field': True,
            'use_role_assignment': True,
            'use_ref_velocity': True,
        },
        'save_interval': 25,
    },
    'optimized': {
        'description': '优化奖励参数（主实验）',
        'num_episodes': 50,
        'seed': 42,
        'reward': {
            'capture_reward': 10.0,
            'chase_reward_coeff': 0.1,
            'escape_reward_coeff': 1.0,
            'alignment_reward_coeff': 0.2,
            'safe_penalty_coeff': 0.3,
            'obstacle_interior_penalty': 0.5,
        },
        'ablation': {
            'use_density_field': True,
            'use_role_assignment': True,
            'use_ref_velocity': True,
        },
        'save_interval': 25,
    },
    'curriculum': {
        'description': 'curriculum training with staged coordination',
        'num_episodes': 300,
        'seed': 42,
        'save_interval': 50,
        'curriculum': {
            'stages': [
                {
                    'name': 'pursuit_avoidance',
                    'until_fraction': 0.30,
                    'reward': {
                        'capture_reward': 12.0,
                        'chase_reward_coeff': 0.6,
                        'escape_reward_coeff': 0.2,
                        'alignment_reward_coeff': 0.2,
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
                        'alignment_reward_coeff': 0.5,
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
        },
    },
    'ablation_no_density': {
        'description': '消融：无密度场分配',
        'num_episodes': 50,
        'seed': 42,
        'reward': {
            'capture_reward': 10.0,
            'chase_reward_coeff': 0.1,
            'escape_reward_coeff': 1.0,
            'alignment_reward_coeff': 0.2,
            'safe_penalty_coeff': 0.3,
            'obstacle_interior_penalty': 0.5,
        },
        'ablation': {
            'use_density_field': False,
            'use_role_assignment': True,
            'use_ref_velocity': True,
        },
        'save_interval': 25,
    },
    'ablation_no_role': {
        'description': '消融：无角色分配',
        'num_episodes': 50,
        'seed': 42,
        'reward': {
            'capture_reward': 10.0,
            'chase_reward_coeff': 0.1,
            'escape_reward_coeff': 1.0,
            'alignment_reward_coeff': 0.2,
            'safe_penalty_coeff': 0.3,
            'obstacle_interior_penalty': 0.5,
        },
        'ablation': {
            'use_density_field': True,
            'use_role_assignment': False,
            'use_ref_velocity': True,
        },
        'save_interval': 25,
    },
    'ablation_no_refvel': {
        'description': '消融：无参考速度',
        'num_episodes': 50,
        'seed': 42,
        'reward': {
            'capture_reward': 10.0,
            'chase_reward_coeff': 0.1,
            'escape_reward_coeff': 1.0,
            'alignment_reward_coeff': 0.0,
            'safe_penalty_coeff': 0.3,
            'obstacle_interior_penalty': 0.5,
        },
        'ablation': {
            'use_density_field': True,
            'use_role_assignment': True,
            'use_ref_velocity': False,
        },
        'save_interval': 25,
    },
}


def resolve_curriculum_stage(curriculum, episode, total_episodes):
    if not curriculum:
        return None
    progress = episode / max(total_episodes, 1)
    stages = curriculum.get('stages', [])
    for stage in stages:
        if progress <= stage.get('until_fraction', 1.0):
            return stage
    return stages[-1] if stages else None


def run_experiment(exp_name):
    """运行单个实验"""
    import numpy as np
    import torch
    import csv
    from MultiTargetEnv import MultiTarEnv, set_global_seeds
    from MATD3 import MATD3Agent, save_training_checkpoint, load_training_checkpoint
    from replaybuffer import ReplayBuffer

    config = EXPERIMENT_CONFIGS[exp_name]
    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    models_dir = os.path.join(exp_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print(f"实验: {exp_name} — {config['description']}", flush=True)
    print(f"训练轮数: {config['num_episodes']}, 种子: {config['seed']}", flush=True)
    print(f"输出目录: {exp_dir}", flush=True)
    print("=" * 70, flush=True)

    seed = config['seed']
    set_global_seeds(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    curriculum = config.get('curriculum')
    abl = config.get('ablation', {
        'use_density_field': True,
        'use_role_assignment': True,
        'use_ref_velocity': True,
    })
    env = MultiTarEnv(
        length=2.0, num_obstacle=5, num_hunters=6, num_targets=2,
        h_actor_dim=32, t_actor_dim=36, action_dim=2,
        visualize_lasers=False,
        use_density_field=abl['use_density_field'],
        use_role_assignment=abl['use_role_assignment'],
        use_ref_velocity=abl['use_ref_velocity'],
    )

    # 应用奖励参数
    rw = config.get('reward')
    if rw:
        env.capture_reward = rw['capture_reward']
        env.chase_reward_coeff = rw['chase_reward_coeff']
        env.escape_reward_coeff = rw['escape_reward_coeff']
        env.alignment_reward_coeff = rw['alignment_reward_coeff']
        env.safe_penalty_coeff = rw['safe_penalty_coeff']
        env.obstacle_interior_penalty = rw['obstacle_interior_penalty']

    num_episodes = config['num_episodes']
    max_steps = 150
    save_interval = config['save_interval']

    hunters = [MATD3Agent(obs_dim=32, action_dim=2, lr=1e-3, gamma=0.95,
                          tau=0.01, noise_std=0.005, device=device,
                          iforthogonalize=True, noise_clip=0.01, a_max=0.01)
               for _ in range(env.num_hunters)]

    targets = [MATD3Agent(obs_dim=36, action_dim=2, lr=1e-3, gamma=0.95,
                          tau=0.01, noise_std=0.1, device=device,
                          iforthogonalize=True, noise_clip=0.01, a_max=0.01)
               for _ in range(env.num_targets)]

    h_buffer = ReplayBuffer(max_size=10000, obs_dim=32, action_dim=2)
    t_buffer = ReplayBuffer(max_size=10000, obs_dim=36, action_dim=2)

    # Checkpoint 续训支持
    checkpoint_path = os.path.join(exp_dir, 'checkpoint.pt')
    csv_path = os.path.join(exp_dir, 'rewards.csv')
    start_episode = 1
    update_counter = 0
    best_score = -float('inf')

    if os.path.exists(checkpoint_path):
        # 从checkpoint恢复
        print(f"发现checkpoint，恢复训练...", flush=True)
        start_episode, update_counter, best_score = load_training_checkpoint(
            checkpoint_path, hunters, targets)
        start_episode += 1  # 从下一轮开始
        print(f"  从第 {start_episode} 轮继续 (update_counter={update_counter}, best={best_score:.1f})", flush=True)
        # CSV追加模式（不覆盖已有数据）
    else:
        # 全新训练，写CSV header
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["episode", "stage", "steps", "capture_success",
                         "total_reward_hunters", "total_reward_targets",
                         "avg_chase_reward", "avg_capture_reward",
                         "avg_escape_reward", "avg_alignment_reward",
                         "avg_critic_loss", "avg_actor_loss"])

    # 随机预热：填充 buffer 后再开始训练
    if start_episode == 1:
        warmup_steps = 2000
        print(f"随机预热中 ({warmup_steps} steps)...", flush=True)
        h_obs, t_obs = env.reset()
        for _ in range(warmup_steps):
            ha = [np.random.uniform(-0.01, 0.01, 2) for _ in range(env.num_hunters)]
            ta = [np.random.uniform(-0.01, 0.01, 2) for _ in range(env.num_targets)]
            h_next, t_next, rewards, dones, ri = env.step(ha + ta)
            rh = rewards[:env.num_hunters]
            rt = rewards[env.num_hunters:]
            for i in range(env.num_hunters):
                h_buffer.store_transition(h_obs[i], ha[i], rh[i], h_next[i], dones[i])
            for i in range(env.num_targets):
                t_buffer.store_transition(t_obs[i], ta[i], rt[i], t_next[i], dones[env.num_hunters + i])
            if any(dones):
                h_obs, t_obs = env.reset()
            else:
                h_obs, t_obs = h_next, t_next
        print(f"预热完成: h_buffer={h_buffer.size()}, t_buffer={t_buffer.size()}", flush=True)

    for episode in tqdm(range(start_episode, num_episodes + 1), desc=f"[{exp_name}]",
                        initial=start_episode - 1, total=num_episodes, ncols=100):
        stage_name = 'default'
        stage = resolve_curriculum_stage(curriculum, episode, num_episodes)
        if stage:
            env.configure_training_phase(
                stage_name=stage.get('name'),
                reward_config=stage.get('reward'),
                ablation_config=stage.get('ablation'),
            )
            stage_name = stage.get('name', 'default')

        h_obs, t_obs = env.reset()
        ep_rh = np.zeros(env.num_hunters)
        ep_rt = np.zeros(env.num_targets)
        done = False
        step = 0
        ep_chase, ep_capture, ep_escape, ep_align = [], [], [], []
        ep_closs, ep_aloss = [], []
        capture = False
        escape = False

        while not done and step < max_steps:
            ha = [h.select_action(h_obs[i]) for i, h in enumerate(hunters)]
            ta = [t.select_action(t_obs[i]) for i, t in enumerate(targets)]
            h_next, t_next, rewards, dones, ri = env.step(ha + ta)

            ep_chase.append(np.mean(ri['chase_rewards']))
            ep_capture.append(np.mean(ri['capture_rewards']))
            ep_escape.append(np.mean(ri['escape_rewards']))
            ep_align.append(np.mean(ri['alignment_rewards']))

            rh = rewards[:env.num_hunters]
            rt = rewards[env.num_hunters:]

            for i in range(env.num_hunters):
                h_buffer.store_transition(h_obs[i], ha[i], rh[i], h_next[i], dones[i])
            for i in range(env.num_targets):
                t_buffer.store_transition(t_obs[i], ta[i], rt[i], t_next[i], dones[env.num_hunters + i])

            ep_rh += rh
            ep_rt += rt
            h_obs, t_obs = h_next, t_next
            done = any(dones)
            if ri.get('capture_happened', False):
                capture = True
            if ri.get('escape_happened', False):
                escape = True
            step += 1

            update_counter += 1
            if update_counter % 10 == 0:
                if h_buffer.size() >= 1024:
                    for _ in range(5):
                        batch = h_buffer.sample(256)
                        for h in hunters:
                            losses = h.update(batch)
                            if losses:
                                ep_closs.append(losses[0])
                                if losses[1] is not None:
                                    ep_aloss.append(losses[1])
                if t_buffer.size() >= 1024:
                    for _ in range(5):
                        batch = t_buffer.sample(256)
                        for t in targets:
                            t.update(batch)

        total_rh = ep_rh.sum()
        total_rt = ep_rt.sum()

        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([episode, stage_name, step, int(capture),
                         f"{total_rh:.4f}", f"{total_rt:.4f}",
                         f"{np.mean(ep_chase):.4f}" if ep_chase else "0",
                         f"{np.mean(ep_capture):.4f}" if ep_capture else "0",
                         f"{np.mean(ep_escape):.4f}" if ep_escape else "0",
                         f"{np.mean(ep_align):.4f}" if ep_align else "0",
                         f"{np.mean(ep_closs):.6f}" if ep_closs else "0",
                         f"{np.mean(ep_aloss):.6f}" if ep_aloss else "0"])

        cap_str = " CAPTURED" if capture else (" ESCAPED" if escape else "")
        if episode % 50 == 0 or capture or escape:
            tqdm.write(f"  Ep {episode}/{num_episodes}  stage={stage_name}  steps={step}  "
                  f"H={total_rh:.1f}  T={total_rt:.1f}{cap_str}")

        # 定期保存checkpoint
        if episode % save_interval == 0:
            ckpt_dir = os.path.join(models_dir, f'ep_{episode}')
            os.makedirs(ckpt_dir, exist_ok=True)
            for i, h in enumerate(hunters):
                h.save_model(ckpt_dir, i, 'hunter')
            for i, t in enumerate(targets):
                t.save_model(ckpt_dir, i, 'target')

        # 保存最佳
        if total_rh > best_score:
            best_score = total_rh
            best_dir = os.path.join(models_dir, 'best')
            os.makedirs(best_dir, exist_ok=True)
            for i, h in enumerate(hunters):
                h.save_model(best_dir, i, 'hunter')
            for i, t in enumerate(targets):
                t.save_model(best_dir, i, 'target')

        # 每50轮保存checkpoint（用于断点续训）
        if episode % 50 == 0:
            save_training_checkpoint(checkpoint_path, episode, hunters, targets,
                                     update_counter, best_score)

    env.close()

    # 训练正常完成，删除checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("  Checkpoint cleaned up (training complete)", flush=True)

    print(f"\n实验 {exp_name} 完成! CSV: {csv_path}", flush=True)

    # 自动生成训练曲线
    from paper_plots import plot_training_curves, plot_reward_components
    plot_training_curves(csv_path, os.path.join(exp_dir, 'plots'))
    plot_reward_components(csv_path, os.path.join(exp_dir, 'plots'))

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="论文实验运行脚本")
    parser.add_argument('--experiment', type=str, default=None,
                        choices=list(EXPERIMENT_CONFIGS.keys()),
                        help='运行指定实验')
    parser.add_argument('--all', action='store_true', help='运行全部实验')
    parser.add_argument('--plots', action='store_true', help='仅生成图表')
    args = parser.parse_args()

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    if args.plots:
        from paper_plots import generate_all_paper_figures
        generate_all_paper_figures(EXPERIMENTS_DIR)
        return

    if args.experiment:
        run_experiment(args.experiment)
    elif args.all:
        exp_order = ['baseline', 'optimized', 'curriculum', 'ablation_no_density',
                      'ablation_no_role', 'ablation_no_refvel']
        for name in exp_order:
            run_experiment(name)
        # 生成汇总图表
        from paper_plots import generate_all_paper_figures
        generate_all_paper_figures(EXPERIMENTS_DIR)
    else:
        parser.print_help()
        print("\n可用实验:")
        for name, cfg in EXPERIMENT_CONFIGS.items():
            print(f"  {name:25s} — {cfg['description']}")


if __name__ == '__main__':
    main()
