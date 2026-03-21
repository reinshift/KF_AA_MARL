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
        'num_episodes': 250,
        'max_steps': 250,
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
        'description': '优化奖励参数',
        'num_episodes': 250,
        'max_steps': 250,
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
        'description': 'curriculum training with balanced coverage and gated coordination',
        'num_episodes': 250,
        'seed': 42,
        'max_steps': 250,
        'save_interval': 50,
        'curriculum': {
            'stages': [
                {
                    'name': 'pursuit_avoidance',
                    'until_fraction': 0.25,
                    'reward': {
                        'capture_reward': 15.0,
                        'team_capture_bonus': 1.5,
                        'chase_reward_coeff': 2.0,
                        'blocked_chase_reward_coeff': 0.0,
                        'stuck_penalty_coeff': 0.0,
                        'containment_progress_reward_coeff': 0.0,
                        'containment_quality_reward_coeff': 0.0,
                        'chaser_slot_reward_coeff': 0.0,
                        'chaser_side_balance_reward_coeff': 0.0,
                        'interceptor_quality_reward_coeff': 0.0,
                        'escape_reward_coeff': 0.05,
                        'alignment_reward_coeff': 0.25,
                        'safe_penalty_coeff': 0.4,
                        'obstacle_interior_penalty': 0.8,
                        'obstacle_proximity_penalty_coeff': 0.7,
                        'hunter_time_penalty_coeff': 0.003,
                        'full_capture_outcome_reward': 8.0,
                        'partial_capture_outcome_reward': 0.5,
                        'timeout_outcome_reward': -0.3,
                        'escape_outcome_reward': -4.0,
                        'target_full_capture_outcome_reward': -3.0,
                        'target_partial_capture_outcome_reward': -0.3,
                        'target_timeout_outcome_reward': 0.0,
                        'target_escape_outcome_reward': 4.0,
                        'distance_threshold': 0.018,
                    },
                    'ablation': {
                        'use_density_field': False,
                        'use_role_assignment': False,
                        'use_ref_velocity': True,
                    },
                    'mechanism': {
                        'assignment_escape_pressure_coeff': 0.0,
                        'density_underloaded_priority': 0.8,
                        'density_over_assignment_penalty': 0.4,
                        'assignment_density_value_weight': 0.7,
                        'assignment_target_proximity_weight': 2.0,
                        'assignment_slot_proximity_weight': 0.8,
                        'assignment_switch_penalty': 0.15,
                        'assignment_target_inertia_bonus': 0.10,
                        'assignment_slot_inertia_bonus': 0.05,
                        'max_interceptors_per_target': 0,
                        'map_refresh_interval': 25,
                        'randomize_exit_zone': False,
                        'target_obs_include_hunters': False,
                        'target_escape_sector_include_hunters': False,
                        'target_hunter_repulsion_scale': 0.0,
                        'target_hunter_contact_penalty_coeff': 0.0,
                        'target_ref_hunter_perception_range': 0.0,
                    },
                    'update': {
                        'hunter_repeats': 5,
                        'target_repeats': 4,
                        'target_train_enabled': True,
                        'hunter_rehearsal_fraction': 0.0,
                        'target_rehearsal_fraction': 0.0,
                    },
                },
                {
                    'name': 'balanced_assignment',
                    'until_fraction': 0.55,
                    'reward': {
                        'capture_reward': 15.0,
                        'team_capture_bonus': 2.5,
                        'chase_reward_coeff': 1.6,
                        'blocked_chase_reward_coeff': 0.25,
                        'stuck_penalty_coeff': 0.05,
                        'containment_progress_reward_coeff': 0.30,
                        'containment_quality_reward_coeff': 0.10,
                        'chaser_slot_reward_coeff': 0.20,
                        'chaser_side_balance_reward_coeff': 0.10,
                        'interceptor_quality_reward_coeff': 0.0,
                        'escape_reward_coeff': 0.10,
                        'alignment_reward_coeff': 0.35,
                        'safe_penalty_coeff': 0.4,
                        'obstacle_interior_penalty': 0.7,
                        'obstacle_proximity_penalty_coeff': 0.75,
                        'hunter_time_penalty_coeff': 0.0035,
                        'full_capture_outcome_reward': 9.0,
                        'partial_capture_outcome_reward': 1.2,
                        'timeout_outcome_reward': -0.5,
                        'escape_outcome_reward': -4.5,
                        'target_full_capture_outcome_reward': -3.2,
                        'target_partial_capture_outcome_reward': -0.6,
                        'target_timeout_outcome_reward': 0.0,
                        'target_escape_outcome_reward': 4.5,
                        'distance_threshold': 0.016,
                    },
                    'ablation': {
                        'use_density_field': True,
                        'use_role_assignment': False,
                        'use_ref_velocity': True,
                    },
                    'mechanism': {
                        'assignment_escape_pressure_coeff': 0.35,
                        'density_underloaded_priority': 1.6,
                        'density_over_assignment_penalty': 1.2,
                        'assignment_density_value_weight': 0.9,
                        'assignment_target_proximity_weight': 2.8,
                        'assignment_slot_proximity_weight': 1.3,
                        'assignment_switch_penalty': 0.35,
                        'assignment_target_inertia_bonus': 0.30,
                        'assignment_slot_inertia_bonus': 0.15,
                        'max_interceptors_per_target': 0,
                        'map_refresh_interval': 18,
                        'randomize_exit_zone': False,
                        'target_obs_include_hunters': True,
                        'target_escape_sector_include_hunters': True,
                        'target_hunter_repulsion_scale': 0.35,
                        'target_hunter_contact_penalty_coeff': 0.35,
                        'target_ref_hunter_perception_range': 0.2,
                    },
                    'update': {
                        'hunter_repeats': 5,
                        'target_repeats': 3,
                        'target_train_enabled': True,
                        'hunter_rehearsal_fraction': 0.15,
                        'target_rehearsal_fraction': 0.10,
                    },
                },
                {
                    'name': 'escape_guidance',
                    'until_fraction': 0.80,
                    'reward': {
                        'capture_reward': 20.0,
                        'team_capture_bonus': 3.0,
                        'chase_reward_coeff': 1.3,
                        'blocked_chase_reward_coeff': 0.50,
                        'stuck_penalty_coeff': 0.10,
                        'containment_progress_reward_coeff': 0.55,
                        'containment_quality_reward_coeff': 0.20,
                        'chaser_slot_reward_coeff': 0.35,
                        'chaser_side_balance_reward_coeff': 0.18,
                        'interceptor_quality_reward_coeff': 0.0,
                        'escape_reward_coeff': 0.10,
                        'alignment_reward_coeff': 0.45,
                        'safe_penalty_coeff': 0.35,
                        'obstacle_interior_penalty': 0.6,
                        'obstacle_proximity_penalty_coeff': 0.85,
                        'hunter_time_penalty_coeff': 0.004,
                        'full_capture_outcome_reward': 10.0,
                        'partial_capture_outcome_reward': 1.8,
                        'timeout_outcome_reward': -0.8,
                        'escape_outcome_reward': -5.0,
                        'target_full_capture_outcome_reward': -3.6,
                        'target_partial_capture_outcome_reward': -0.9,
                        'target_timeout_outcome_reward': -0.1,
                        'target_escape_outcome_reward': 5.0,
                        'distance_threshold': 0.014,
                    },
                    'ablation': {
                        'use_density_field': True,
                        'use_role_assignment': False,
                        'use_ref_velocity': True,
                    },
                    'mechanism': {
                        'assignment_escape_pressure_coeff': 0.55,
                        'density_underloaded_priority': 1.8,
                        'density_over_assignment_penalty': 1.4,
                        'assignment_density_value_weight': 1.0,
                        'assignment_target_proximity_weight': 3.0,
                        'assignment_slot_proximity_weight': 1.5,
                        'assignment_switch_penalty': 0.45,
                        'assignment_target_inertia_bonus': 0.35,
                        'assignment_slot_inertia_bonus': 0.20,
                        'max_interceptors_per_target': 0,
                        'map_refresh_interval': 12,
                        'randomize_exit_zone': True,
                        'target_obs_include_hunters': True,
                        'target_escape_sector_include_hunters': True,
                        'target_hunter_repulsion_scale': 0.75,
                        'target_hunter_contact_penalty_coeff': 0.75,
                        'target_ref_hunter_perception_range': 0.35,
                    },
                    'update': {
                        'hunter_repeats': 5,
                        'target_repeats': 4,
                        'target_train_enabled': True,
                        'hunter_rehearsal_fraction': 0.25,
                        'target_rehearsal_fraction': 0.10,
                    },
                },
                {
                    'name': 'full_coordination',
                    'until_fraction': 1.00,
                    'reward': {
                        'capture_reward': 16.0,
                        'team_capture_bonus': 4.0,
                        'chase_reward_coeff': 1.10,
                        'blocked_chase_reward_coeff': 0.55,
                        'stuck_penalty_coeff': 0.15,
                        'containment_progress_reward_coeff': 0.80,
                        'containment_quality_reward_coeff': 0.30,
                        'chaser_slot_reward_coeff': 0.45,
                        'chaser_side_balance_reward_coeff': 0.25,
                        'interceptor_quality_reward_coeff': 1.20,
                        'escape_reward_coeff': 0.15,
                        'alignment_reward_coeff': 0.5,
                        'safe_penalty_coeff': 0.20,
                        'obstacle_interior_penalty': 0.6,
                        'obstacle_proximity_penalty_coeff': 0.90,
                        'hunter_time_penalty_coeff': 0.005,
                        'full_capture_outcome_reward': 12.0,
                        'partial_capture_outcome_reward': 2.5,
                        'timeout_outcome_reward': -1.0,
                        'escape_outcome_reward': -6.0,
                        'target_full_capture_outcome_reward': -4.0,
                        'target_partial_capture_outcome_reward': -1.2,
                        'target_timeout_outcome_reward': -0.2,
                        'target_escape_outcome_reward': 6.0,
                        'distance_threshold': 0.012,
                    },
                    'ablation': {
                        'use_density_field': True,
                        'use_role_assignment': True,
                        'use_ref_velocity': True,
                    },
                    'mechanism': {
                        'assignment_escape_pressure_coeff': 0.70,
                        'density_underloaded_priority': 2.0,
                        'density_over_assignment_penalty': 1.5,
                        'assignment_density_value_weight': 1.0,
                        'assignment_target_proximity_weight': 3.2,
                        'assignment_slot_proximity_weight': 1.7,
                        'assignment_switch_penalty': 0.55,
                        'assignment_target_inertia_bonus': 0.40,
                        'assignment_slot_inertia_bonus': 0.24,
                        'min_group_size_for_interceptor': 3,
                        'max_interceptors_per_target': 1,
                        'min_target_speed_for_interceptor': 0.03,
                        'min_interceptor_distance': 0.14,
                        'max_interceptor_distance': 0.28,
                        'interceptor_prediction_steps': 3,
                        'interceptor_persistence_bonus': 0.18,
                        'intercept_projection_clearance': 0.05,
                        'map_refresh_interval': 8,
                        'randomize_exit_zone': True,
                        'target_obs_include_hunters': True,
                        'target_escape_sector_include_hunters': True,
                        'target_hunter_repulsion_scale': 1.0,
                        'target_hunter_contact_penalty_coeff': 1.0,
                        'target_ref_hunter_perception_range': 0.5,
                    },
                    'update': {
                        'hunter_repeats': 4,
                        'target_repeats': 3,
                        'target_train_enabled': True,
                        'hunter_rehearsal_fraction': 0.35,
                        'target_rehearsal_fraction': 0.15,
                    },
                },
            ],
        },
    },
    'ablation_no_density': {
        'description': '消融：无密度场分配',
        'num_episodes': 250,
        'max_steps': 250,
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
        'num_episodes': 250,
        'max_steps': 250,
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
        'num_episodes': 250,
        'max_steps': 250,
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
        env.configure_training_phase(reward_config=rw)
    if curriculum:
        initial_stage = resolve_curriculum_stage(curriculum, 1, config['num_episodes'])
        if initial_stage:
            env.configure_training_phase(
                stage_name=initial_stage.get('name'),
                reward_config=initial_stage.get('reward'),
                ablation_config=initial_stage.get('ablation'),
                mechanism_config=initial_stage.get('mechanism'),
            )

    num_episodes = config['num_episodes']
    max_steps = config.get('max_steps', 150)
    save_interval = config['save_interval']

    hunters = [MATD3Agent(obs_dim=env.h_actor_dim, action_dim=2, lr=1e-3, gamma=0.95,
                          tau=0.01, noise_std=0.005, device=device,
                          iforthogonalize=True, noise_clip=0.01, a_max=0.01)
               for _ in range(env.num_hunters)]

    targets = [MATD3Agent(obs_dim=env.t_actor_dim, action_dim=2, lr=1e-3, gamma=0.95,
                          tau=0.01, noise_std=0.1, device=device,
                          iforthogonalize=True, noise_clip=0.01, a_max=0.01)
               for _ in range(env.num_targets)]

    h_buffer = ReplayBuffer(
        max_size=10000,
        obs_dim=env.h_actor_dim,
        action_dim=2,
        rehearsal_size=2500 if curriculum else 0,
    )
    t_buffer = ReplayBuffer(
        max_size=10000,
        obs_dim=env.t_actor_dim,
        action_dim=2,
        rehearsal_size=1500 if curriculum else 0,
    )

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
                         "outcome_code", "captured_target_count", "escaped_target_count",
                         "total_reward_hunters", "total_reward_targets",
                          "avg_chase_reward", "avg_capture_reward",
                          "avg_escape_reward", "avg_alignment_reward",
                         "avg_gap_reward", "avg_stuck_penalty", "avg_blocked_chase",
                         "avg_containment_reward", "avg_role_geometry_reward", "avg_escape_bandwidth",
                         "avg_critic_loss", "avg_actor_loss",
                         "avg_active_targets", "avg_min_group_size",
                         "avg_max_group_size", "avg_group_size_std",
                         "avg_interceptors"])

    # 随机预热：填充 buffer 后再开始训练
    if start_episode == 1:
        warmup_steps = 2000
        print(f"随机预热中 ({warmup_steps} steps)...", flush=True)
        h_obs, t_obs = env.reset()
        for _ in range(warmup_steps):
            ha = [np.random.uniform(-0.01, 0.01, 2) for _ in range(env.num_hunters)]
            active_target_mask = [env._is_target_active(target) for target in env.targets]
            ta = [
                np.random.uniform(-0.01, 0.01, 2) if active_target_mask[i] else np.zeros(2, dtype=float)
                for i in range(env.num_targets)
            ]
            h_next, t_next, rewards, dones, ri = env.step(ha + ta)
            rh = rewards[:env.num_hunters]
            rt = rewards[env.num_hunters:]
            for i in range(env.num_hunters):
                h_buffer.store_transition(h_obs[i], ha[i], rh[i], h_next[i], dones[i])
            for i in range(env.num_targets):
                if active_target_mask[i]:
                    t_buffer.store_transition(t_obs[i], ta[i], rt[i], t_next[i], dones[env.num_hunters + i])
            if ri.get('episode_terminal', any(dones)):
                h_obs, t_obs = env.reset()
            else:
                h_obs, t_obs = h_next, t_next
        print(f"预热完成: h_buffer={h_buffer.size()}, t_buffer={t_buffer.size()}", flush=True)

    for episode in tqdm(range(start_episode, num_episodes + 1), desc=f"[{exp_name}]",
                        initial=start_episode - 1, total=num_episodes, ncols=100):
        stage_name = 'default'
        update_cfg = {
            'hunter_repeats': 5,
            'target_repeats': 5,
            'target_train_enabled': True,
            'hunter_rehearsal_fraction': 0.0,
            'target_rehearsal_fraction': 0.0,
        }
        stage = resolve_curriculum_stage(curriculum, episode, num_episodes)
        if stage:
            env.configure_training_phase(
                stage_name=stage.get('name'),
                reward_config=stage.get('reward'),
                ablation_config=stage.get('ablation'),
                mechanism_config=stage.get('mechanism'),
            )
            stage_name = stage.get('name', 'default')
            update_cfg.update(stage.get('update', {}))

        h_obs, t_obs = env.reset()
        ep_rh = np.zeros(env.num_hunters)
        ep_rt = np.zeros(env.num_targets)
        done = False
        step = 0
        ep_chase, ep_capture, ep_escape, ep_align = [], [], [], []
        ep_gap, ep_stuck, ep_blocked = [], [], []
        ep_containment, ep_role_geom, ep_escape_bandwidth = [], [], []
        ep_active_targets, ep_min_group, ep_max_group, ep_group_std, ep_interceptors = [], [], [], [], []
        ep_closs, ep_aloss = [], []
        capture = False
        escape = False
        outcome_code = None
        captured_target_count = 0
        escaped_target_count = 0

        while not done and step < max_steps:
            ha = [h.select_action(h_obs[i]) for i, h in enumerate(hunters)]
            active_target_mask = [env._is_target_active(target) for target in env.targets]
            ta = [
                t.select_action(t_obs[i]) if active_target_mask[i] else np.zeros(2, dtype=float)
                for i, t in enumerate(targets)
            ]
            h_next, t_next, rewards, dones, ri = env.step(ha + ta)

            ep_chase.append(np.mean(ri['chase_rewards']))
            ep_capture.append(np.mean(ri['capture_rewards']))
            ep_escape.append(np.mean(ri['escape_rewards']))
            ep_align.append(np.mean(ri['alignment_rewards']))
            ep_gap.append(ri.get('avg_gap_reward', 0.0))
            ep_stuck.append(ri.get('avg_stuck_penalty', 0.0))
            ep_blocked.append(ri.get('blocked_chase_ratio', 0.0))
            ep_containment.append(ri.get('avg_containment_reward', 0.0))
            ep_role_geom.append(ri.get('avg_role_geometry_reward', 0.0))
            ep_escape_bandwidth.append(ri.get('avg_escape_bandwidth', 1.0))
            ep_active_targets.append(ri.get('active_target_count', env.num_targets))
            ep_min_group.append(ri.get('min_group_size', 0))
            ep_max_group.append(ri.get('max_group_size', 0))
            ep_group_std.append(ri.get('group_size_std', 0.0))
            ep_interceptors.append(ri.get('interceptor_count', 0))

            timed_out = (step + 1 >= max_steps) and not ri.get('episode_terminal', any(dones))
            if timed_out:
                rewards, timeout_info = env.finalize_timeout_outcome(rewards)
                ri.update(timeout_info)
                dones = [True] * (env.num_hunters + env.num_targets)

            rh = rewards[:env.num_hunters]
            rt = rewards[env.num_hunters:]

            for i in range(env.num_hunters):
                h_buffer.store_transition(h_obs[i], ha[i], rh[i], h_next[i], dones[i])
            for i in range(env.num_targets):
                if active_target_mask[i]:
                    t_buffer.store_transition(t_obs[i], ta[i], rt[i], t_next[i], dones[env.num_hunters + i])

            ep_rh += rh
            ep_rt += rt
            h_obs, t_obs = h_next, t_next
            done = ri.get('episode_terminal', any(dones)) or timed_out
            if ri.get('capture_happened', False):
                capture = True
            if ri.get('escape_happened', False):
                escape = True
            if ri.get('outcome_code') is not None:
                outcome_code = ri.get('outcome_code')
            captured_target_count = max(captured_target_count, int(ri.get('captured_target_count', 0)))
            escaped_target_count = max(escaped_target_count, int(ri.get('escaped_target_count', 0)))
            step += 1

            update_counter += 1
            if update_counter % 10 == 0:
                if h_buffer.size() >= 1024:
                    for _ in range(update_cfg.get('hunter_repeats', 5)):
                        batch = h_buffer.sample(
                            256,
                            rehearsal_fraction=update_cfg.get('hunter_rehearsal_fraction', 0.0),
                        )
                        for h in hunters:
                            losses = h.update(batch)
                            if losses:
                                ep_closs.append(losses[0])
                                if losses[1] is not None:
                                    ep_aloss.append(losses[1])
                if update_cfg.get('target_train_enabled', True) and t_buffer.size() >= 1024:
                    for _ in range(update_cfg.get('target_repeats', 5)):
                        batch = t_buffer.sample(
                            256,
                            rehearsal_fraction=update_cfg.get('target_rehearsal_fraction', 0.0),
                        )
                        for t in targets:
                            t.update(batch)

        total_rh = ep_rh.sum()
        total_rt = ep_rt.sum()
        if outcome_code is None:
            outcome_code = env._get_outcome_code(timed_out=(not capture and not escape))
        outcome_priority = {2: 1000.0, 1: 200.0, 0: 0.0, -1: -200.0}.get(outcome_code, 0.0)
        model_score = total_rh + outcome_priority

        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([episode, stage_name, step, int(capture),
                         outcome_code if outcome_code is not None else "",
                         captured_target_count, escaped_target_count,
                         f"{total_rh:.4f}", f"{total_rt:.4f}",
                         f"{np.mean(ep_chase):.4f}" if ep_chase else "0",
                         f"{np.mean(ep_capture):.4f}" if ep_capture else "0",
                         f"{np.mean(ep_escape):.4f}" if ep_escape else "0",
                         f"{np.mean(ep_align):.4f}" if ep_align else "0",
                         f"{np.mean(ep_gap):.4f}" if ep_gap else "0",
                         f"{np.mean(ep_stuck):.4f}" if ep_stuck else "0",
                         f"{np.mean(ep_blocked):.4f}" if ep_blocked else "0",
                         f"{np.mean(ep_containment):.4f}" if ep_containment else "0",
                         f"{np.mean(ep_role_geom):.4f}" if ep_role_geom else "0",
                         f"{np.mean(ep_escape_bandwidth):.4f}" if ep_escape_bandwidth else "1",
                         f"{np.mean(ep_closs):.6f}" if ep_closs else "0",
                         f"{np.mean(ep_aloss):.6f}" if ep_aloss else "0",
                         f"{np.mean(ep_active_targets):.4f}" if ep_active_targets else "0",
                         f"{np.mean(ep_min_group):.4f}" if ep_min_group else "0",
                         f"{np.mean(ep_max_group):.4f}" if ep_max_group else "0",
                         f"{np.mean(ep_group_std):.4f}" if ep_group_std else "0",
                         f"{np.mean(ep_interceptors):.4f}" if ep_interceptors else "0"])

        if outcome_code == 2:
            cap_str = " ALL_CAPTURED"
        elif outcome_code == 1:
            cap_str = " PARTIAL_TIMEOUT"
        elif outcome_code == -1:
            cap_str = " ESCAPED"
        else:
            cap_str = ""
        if episode % 50 == 0 or capture or escape or outcome_code == 1:
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
        if model_score > best_score:
            best_score = model_score
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
