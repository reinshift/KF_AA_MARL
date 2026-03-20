"""
论文级绘图工具 (Paper-quality Plotting Tools)

为论文生成标准化的实验图表，包括：
- 训练曲线（奖励、loss、捕获率、步数）
- 分奖励项随训练变化
- 消融实验对比图
- 轨迹鸟瞰图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 论文级图表全局配置
def setup_paper_style():
    """设置论文级图表样式"""
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 14
    rcParams['legend.fontsize'] = 11
    rcParams['xtick.labelsize'] = 11
    rcParams['ytick.labelsize'] = 11
    rcParams['lines.linewidth'] = 1.5
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    rcParams['figure.figsize'] = (8, 5)
    rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']
    rcParams['font.family'] = 'sans-serif'
    rcParams['axes.unicode_minus'] = False

setup_paper_style()


def _smooth(data, window=20):
    """滑动窗口平滑"""
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


def _save_fig(fig, save_dir, name):
    """保存图表为PNG和PDF"""
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{name}.png"), bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, f"{name}.pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.png/pdf")


def plot_training_curves(csv_path, save_dir, window=30):
    """
    绘制训练曲线：奖励、loss、捕获率、步数
    4子图布局
    """
    df = pd.read_csv(csv_path)
    episodes = df['episode'].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Hunter/Target Rewards
    ax = axes[0, 0]
    ax.plot(episodes, df['total_reward_hunters'], alpha=0.2, color='#d62728')
    ax.plot(episodes, _smooth(df['total_reward_hunters'], window), color='#d62728', label='追击者奖励')
    ax.plot(episodes, df['total_reward_targets'], alpha=0.2, color='#2ca02c')
    ax.plot(episodes, _smooth(df['total_reward_targets'], window), color='#2ca02c', label='逃逸者奖励')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Rewards')
    ax.legend()

    # 2. Capture Rate (滑动窗口)
    ax = axes[0, 1]
    capture = df['capture_success'].values.astype(float)
    capture_rate = _smooth(capture, window)
    ax.plot(episodes, capture_rate, color='#1f77b4', linewidth=2)
    ax.fill_between(episodes, 0, capture_rate, alpha=0.15, color='#1f77b4')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Capture Rate')
    ax.set_title(f'Capture Rate (window={window})')
    ax.set_ylim(-0.05, 1.05)

    # 3. Actor/Critic Loss
    ax = axes[1, 0]
    if 'avg_critic_loss' in df.columns:
        critic = df['avg_critic_loss'].values
        actor = df['avg_actor_loss'].values
        mask = critic > 0  # 跳过无loss的early episodes
        if mask.sum() > 0:
            ax.plot(episodes[mask], _smooth(critic[mask], window), color='#ff7f0e', label='评论家损失')
            ax.plot(episodes[mask], _smooth(np.abs(actor[mask]), window), color='#9467bd', label='演员损失绝对值')
            ax.set_yscale('log')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Actor/Critic Loss')
    ax.legend()

    # 4. Episode Length
    ax = axes[1, 1]
    steps = df['steps'].values
    ax.plot(episodes, steps, alpha=0.2, color='#8c564b')
    ax.plot(episodes, _smooth(steps, window), color='#8c564b', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')

    fig.suptitle('Training Progress', fontsize=16, y=1.01)
    fig.tight_layout()
    _save_fig(fig, save_dir, 'fig_training_curves')


def plot_reward_components(csv_path, save_dir, window=30):
    """绘制分奖励项随训练变化"""
    df = pd.read_csv(csv_path)
    episodes = df['episode'].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    components = [
        ('avg_chase_reward', 'Chase Reward', '#d62728', axes[0, 0]),
        ('avg_capture_reward', 'Capture Reward', '#1f77b4', axes[0, 1]),
        ('avg_escape_reward', 'Escape Reward', '#2ca02c', axes[1, 0]),
        ('avg_alignment_reward', 'Alignment Reward', '#ff7f0e', axes[1, 1]),
    ]

    for col, title, color, ax in components:
        if col in df.columns:
            data = df[col].values
            ax.plot(episodes, data, alpha=0.2, color=color)
            ax.plot(episodes, _smooth(data, window), color=color, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(title)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle('Reward Components', fontsize=16, y=1.01)
    fig.tight_layout()
    _save_fig(fig, save_dir, 'fig_reward_components')


def plot_ablation_comparison(csv_paths_dict, save_dir, window=30):
    """
    消融实验对比图

    Args:
        csv_paths_dict: {'实验名': 'csv路径', ...}
        save_dir: 保存目录
    """
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, csv_path) in enumerate(csv_paths_dict.items()):
        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found, skipping {name}")
            continue
        df = pd.read_csv(csv_path)
        episodes = df['episode'].values
        color = colors[idx % len(colors)]

        # 1. Capture Rate
        capture_rate = _smooth(df['capture_success'].values.astype(float), window)
        axes[0].plot(episodes, capture_rate, color=color, label=name, linewidth=2)

        # 2. Hunter Reward
        h_reward = _smooth(df['total_reward_hunters'].values, window)
        axes[1].plot(episodes, h_reward, color=color, label=name, linewidth=2)

        # 3. Episode Steps
        steps = _smooth(df['steps'].values, window)
        axes[2].plot(episodes, steps, color=color, label=name, linewidth=2)

    axes[0].set_title('Capture Rate')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Rate')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()

    axes[1].set_title('Hunter Total Reward')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Reward')
    axes[1].legend()

    axes[2].set_title('Episode Length')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Steps')
    axes[2].legend()

    fig.suptitle('Ablation Study Comparison', fontsize=16, y=1.02)
    fig.tight_layout()
    _save_fig(fig, save_dir, 'fig_ablation_comparison')


def plot_ablation_bar(csv_paths_dict, save_dir, last_n=100):
    """
    消融实验柱状图 — 最后N轮的平均指标对比

    Args:
        csv_paths_dict: {'实验名': 'csv路径', ...}
        last_n: 取最后N轮的平均值
    """
    names = []
    capture_rates = []
    avg_rewards = []
    avg_steps = []

    for name, csv_path in csv_paths_dict.items():
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        tail = df.tail(last_n)
        names.append(name)
        capture_rates.append(tail['capture_success'].mean())
        avg_rewards.append(tail['total_reward_hunters'].mean())
        avg_steps.append(tail['steps'].mean())

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(x, capture_rates, color='#1f77b4', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha='right')
    axes[0].set_ylabel('Capture Rate')
    axes[0].set_title(f'Avg Capture Rate (last {last_n} ep)')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(capture_rates):
        axes[0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=10)

    axes[1].bar(x, avg_rewards, color='#d62728', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha='right')
    axes[1].set_ylabel('Hunter Reward')
    axes[1].set_title(f'Avg Hunter Reward (last {last_n} ep)')

    axes[2].bar(x, avg_steps, color='#2ca02c', alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha='right')
    axes[2].set_ylabel('Steps')
    axes[2].set_title(f'Avg Episode Steps (last {last_n} ep)')

    fig.suptitle('Ablation Study Summary', fontsize=16, y=1.02)
    fig.tight_layout()
    _save_fig(fig, save_dir, 'fig_ablation_bar')


def plot_trajectory(env, save_path, title='Pursuit Trajectory'):
    """
    绘制单回合轨迹鸟瞰图

    Args:
        env: MultiTarEnv实例（已运行完一个episode，history_pos有数据）
        save_path: 保存路径
        title: 图标题
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制边界
    ax.plot([0, env.length, env.length, 0, 0],
            [0, 0, env.length, env.length, 0], 'k-', linewidth=2)

    # 绘制障碍物
    for obs in env.obstacles:
        circle = plt.Circle((obs.position[0], obs.position[1]),
                             obs.radius, color='gray', alpha=0.5)
        ax.add_patch(circle)

    # 绘制Hunter轨迹
    for i, hunter in enumerate(env.hunters):
        if len(hunter.history_pos) > 1:
            traj = np.array(hunter.history_pos)
            # 颜色渐变表示时间
            n = len(traj)
            for j in range(n - 1):
                alpha = 0.3 + 0.7 * j / n
                ax.plot(traj[j:j+2, 0], traj[j:j+2, 1],
                        color='red', alpha=alpha, linewidth=1.5)
            # 起点和终点
            ax.plot(traj[0, 0], traj[0, 1], 'bs', markersize=6)
            ax.plot(traj[-1, 0], traj[-1, 1], 'rx', markersize=8, markeredgewidth=2)

    # 绘制Target轨迹
    for i, target in enumerate(env.targets):
        if len(target.history_pos) > 1:
            traj = np.array(target.history_pos)
            n = len(traj)
            for j in range(n - 1):
                alpha = 0.3 + 0.7 * j / n
                ax.plot(traj[j:j+2, 0], traj[j:j+2, 1],
                        color='green', alpha=alpha, linewidth=1.5)
            ax.plot(traj[0, 0], traj[0, 1], 'bs', markersize=6)
            ax.plot(traj[-1, 0], traj[-1, 1], 'gx', markersize=8, markeredgewidth=2)

    ax.set_xlim(-0.05, env.length + 0.05)
    ax.set_ylim(-0.05, env.length + 0.05)
    ax.set_aspect('equal')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title(title)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='追击者'),
        Line2D([0], [0], color='green', linewidth=2, label='逃逸者'),
        Line2D([0], [0], marker='s', color='blue', linestyle='None', markersize=6, label='起点'),
        Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=8, markeredgewidth=2, label='终点'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved trajectory: {save_path}")


def generate_all_paper_figures(experiments_dir):
    """
    从paper_experiments/目录生成所有论文图表

    Args:
        experiments_dir: paper_experiments/ 目录路径
    """
    figures_dir = os.path.join(experiments_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # 1. 主实验训练曲线
    optimized_csv = os.path.join(experiments_dir, 'optimized', 'rewards.csv')
    if os.path.exists(optimized_csv):
        print("Generating training curves (optimized)...")
        plot_training_curves(optimized_csv, figures_dir)
        plot_reward_components(optimized_csv, figures_dir)

    # 2. 消融实验对比
    ablation_csvs = {}
    exp_names = {
        'baseline': '基线方法',
        'optimized': '优化方法',
        'ablation_no_density': '无密度场',
        'ablation_no_role': '无角色分配',
        'ablation_no_refvel': '无引导速度',
    }
    for folder, label in exp_names.items():
        csv_path = os.path.join(experiments_dir, folder, 'rewards.csv')
        if os.path.exists(csv_path):
            ablation_csvs[label] = csv_path

    if len(ablation_csvs) >= 2:
        print("Generating ablation comparison...")
        plot_ablation_comparison(ablation_csvs, figures_dir)
        plot_ablation_bar(ablation_csvs, figures_dir)

    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        generate_all_paper_figures(sys.argv[1])
    else:
        print("Usage: python paper_plots.py <paper_experiments_dir>")
