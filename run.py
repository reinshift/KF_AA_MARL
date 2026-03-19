#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KF_AA_MARL 统一入口脚本

用法:
    python run.py train                    # 训练
    python run.py train_continue           # 继续训练（需在config.yaml中设置checkpoint）
    python run.py test                     # 测试模型
    python run.py validate                 # 运行验证流水线
    python run.py plot                     # 绘制训练曲线
    python run.py plot --file_path x.csv   # 指定CSV绘图
    python run.py train --config my.yaml   # 使用自定义配置
"""

import os
import sys
import argparse
import yaml

# 将src目录加入路径
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def load_config(config_path):
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_train_args(config, continue_training=False):
    """将config字典转为src/main.py期望的argparse.Namespace"""
    env = config.get('environment', {})
    train = config.get('training', {})
    model = config.get('model', {})

    checkpoint = train.get('checkpoint')
    if continue_training and not checkpoint:
        checkpoint = model.get('path')

    return argparse.Namespace(
        env_length=env.get('env_length', 2.0),
        num_obstacle=env.get('num_obstacles', 5),
        num_hunters=env.get('num_hunters', 6),
        num_targets=env.get('num_targets', 2),
        h_actor_dim=model.get('h_actor_dim', 32),
        t_actor_dim=model.get('t_actor_dim', 33),
        action_dim=model.get('action_dim', 2),
        a_max=train.get('a_max', 0.01),
        ifrender=train.get('ifrender', False),
        visualizelaser=train.get('visualizelaser', False),
        seed=train.get('seed', 10),
        lr=float(train.get('lr', 5e-4)),
        gamma=train.get('gamma', 0.95),
        tau=train.get('tau', 0.01),
        noise_std=train.get('noise_std', 0.005),
        noise_clip=train.get('noise_clip', 0.01),
        buffer_size=train.get('buffer_size', 10000),
        min_buffer_size=train.get('min_buffer_size', 1024),
        num_episodes=train.get('num_episodes', 500),
        max_steps=train.get('max_steps', 150),
        batch_size=train.get('batch_size', 256),
        update_freq=train.get('update_freq', 10),
        update_iterations=train.get('update_iterations', 20),
        save_frequency=train.get('save_frequency', 250),
        score_threshold=train.get('score_threshold', 500),
        iforthogonalize=train.get('iforthogonalize', True),
        iflrdecay=train.get('iflrdecay', False),
        checkpoint=checkpoint,
    )


def run_test(config):
    """运行测试模式：加载模型并可视化"""
    import numpy as np
    import torch
    from MultiTargetEnv import MultiTarEnv, set_global_seeds
    from MATD3 import MATD3Agent

    env_cfg = config.get('environment', {})
    model_cfg = config.get('model', {})
    test_cfg = config.get('test', {})

    model_path = model_cfg.get('path', '')
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        sys.exit(1)

    seed = test_cfg.get('seed', 10)
    set_global_seeds(seed)

    h_actor_dim = model_cfg.get('h_actor_dim', 32)
    t_actor_dim = model_cfg.get('t_actor_dim', 33)
    action_dim = model_cfg.get('action_dim', 2)

    env = MultiTarEnv(
        length=env_cfg.get('env_length', 2.0),
        num_obstacle=env_cfg.get('num_obstacles', 5),
        num_hunters=env_cfg.get('num_hunters', 6),
        num_targets=env_cfg.get('num_targets', 2),
        h_actor_dim=h_actor_dim,
        t_actor_dim=t_actor_dim,
        action_dim=action_dim,
        visualize_lasers=test_cfg.get('visualizelaser', False)
    )

    # 加载模型
    hunters = []
    for i in range(env.num_hunters):
        agent = MATD3Agent(obs_dim=h_actor_dim, action_dim=action_dim,
                           lr=1e-3, gamma=0.95, tau=0.01, noise_std=0.1,
                           a_max=env.a_max, device='cpu')
        agent.load_model(model_path, agent_id=i, agent_type='hunter')
        hunters.append(agent)

    targets = []
    for i in range(env.num_targets):
        agent = MATD3Agent(obs_dim=t_actor_dim, action_dim=action_dim,
                           lr=1e-3, gamma=0.95, tau=0.01, noise_std=0.1,
                           a_max=env.a_max, device='cpu')
        agent.load_model(model_path, agent_id=i, agent_type='target')
        targets.append(agent)

    print(f"模型已加载: {model_path}")

    num_episodes = test_cfg.get('num_episodes', 5)
    max_steps = config.get('training', {}).get('max_steps', 150)
    do_render = test_cfg.get('ifrender', True)

    # 检测target观测维度兼容性
    target_obs_dim = t_actor_dim
    trim_target_obs = False

    for ep in range(num_episodes):
        h_obs, t_obs = env.reset()
        env_t_dim = len(t_obs[0]) if t_obs else t_actor_dim
        trim_target_obs = (target_obs_dim < env_t_dim)
        if trim_target_obs:
            t_obs = [obs[:target_obs_dim] for obs in t_obs]

        print(f"\n--- 测试回合 {ep + 1}/{num_episodes} ---")
        for step in range(max_steps):
            h_actions = [h.select_action(h_obs[i], noise=False) for i, h in enumerate(hunters)]
            t_actions = [t.select_action(t_obs[i], noise=False) for i, t in enumerate(targets)]

            h_obs, t_obs, rewards, dones = env.step(h_actions + t_actions)
            if trim_target_obs:
                t_obs = [obs[:target_obs_dim] for obs in t_obs]

            if do_render:
                env.render()

            if any(dones):
                print(f"  捕获成功! 步数: {step + 1}")
                break
        else:
            print(f"  未捕获, 达到最大步数 {max_steps}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="KF_AA_MARL 多无人机围捕系统")
    parser.add_argument("mode", choices=["train", "train_continue", "test", "validate", "plot"],
                        help="运行模式")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径 (默认: config.yaml)")
    parser.add_argument("--file_path", type=str, default=None,
                        help="CSV文件路径 (仅plot模式)")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "train":
        from main import main as train_main
        train_args = build_train_args(config, continue_training=False)
        train_main(train_args)

    elif args.mode == "train_continue":
        from main import main as train_main
        train_args = build_train_args(config, continue_training=True)
        if not train_args.checkpoint:
            print("错误: 继续训练需要设置 training.checkpoint 或 model.path")
            sys.exit(1)
        train_main(train_args)

    elif args.mode == "test":
        run_test(config)

    elif args.mode == "validate":
        from validation_pipeline import ValidationPipeline
        pipeline = ValidationPipeline(args.config)
        pipeline.run_validation()

    elif args.mode == "plot":
        from plotcurve import plot_rewards
        if args.file_path:
            file_path = args.file_path
        else:
            # 自动查找最新的训练数据
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_train')
            if not os.path.exists(data_dir):
                print("错误: data_train 目录不存在，请先训练或指定CSV路径")
                sys.exit(1)
            folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, f))]
            if not folders:
                print("错误: data_train 目录为空")
                sys.exit(1)
            latest = max(folders, key=os.path.getmtime)
            file_path = os.path.join(latest, 'rewards.csv')

        plot_rewards(file_path, window_size=10)


if __name__ == "__main__":
    main()
