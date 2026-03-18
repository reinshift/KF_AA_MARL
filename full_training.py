"""
完整训练脚本 - 500轮训练验证新功能

使用新实现的功能:
1. 密度场分配机制
2. Target逃逸策略（参考速度和奖励函数）
"""

import numpy as np
import torch
import sys
import os
sys.path.append('src')

from MultiTargetEnv import MultiTarEnv, set_global_seeds
from MATD3 import MATD3Agent
from replaybuffer import ReplayBuffer
import matplotlib.pyplot as plt
from datetime import datetime
import csv

def full_training():
    """运行完整训练"""
    print("=" * 80)
    print("完整训练 - 500轮验证新功能")
    print("=" * 80)
    print()
    
    # 设置参数
    seed = 42
    num_episodes = 500
    max_steps = 150
    
    set_global_seeds(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print()
    
    # 创建环境
    print("1. 创建环境...")
    env = MultiTarEnv(
        length=2.0,
        num_obstacle=5,
        num_hunters=6,
        num_targets=2,
        h_actor_dim=32,
        t_actor_dim=33,  # 注意: 增加了2维（参考速度）
        action_dim=2,
        visualize_lasers=False
    )
    print(f"   - Hunters: {env.num_hunters}")
    print(f"   - Targets: {env.num_targets}")
    print(f"   - 障碍物: {env.num_obstacle}")
    print()
    
    # 初始化agents
    print("2. 初始化agents...")
    hunters = [MATD3Agent(
        obs_dim=32,
        action_dim=2,
        lr=5e-4,
        gamma=0.95,
        tau=0.01,
        noise_std=0.005,
        device=device,
        iforthogonalize=True,
        noise_clip=0.01,
        a_max=0.01,
        if_lr_decay=False,
        total_episodes=num_episodes
    ) for _ in range(env.num_hunters)]
    
    targets = [MATD3Agent(
        obs_dim=33,
        action_dim=2,
        lr=5e-4,
        gamma=0.95,
        tau=0.01,
        noise_std=0.005,
        device=device,
        iforthogonalize=True,
        noise_clip=0.01,
        a_max=0.01,
        if_lr_decay=False,
        total_episodes=num_episodes
    ) for _ in range(env.num_targets)]
    print("   ✓ Agents初始化完成")
    print()
    
    # 初始化replay buffers
    hunters_buffer = ReplayBuffer(max_size=10000, obs_dim=32, action_dim=2)
    targets_buffer = ReplayBuffer(max_size=10000, obs_dim=33, action_dim=2)
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = os.path.join(os.getcwd(), "data_train", f"{timestamp}_full_training")
    os.makedirs(data_dir, exist_ok=True)
    
    # 初始化CSV文件
    rewards_csv_path = os.path.join(data_dir, "rewards.csv")
    with open(rewards_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "total_reward_hunters", "total_reward_targets", 
                        "avg_alignment", "density_changes", "capture_success"])
    
    # 训练统计
    rewards_history = {
        'hunters': [],
        'targets': [],
        'alignment': [],
        'density_changes': [],
        'capture_success': []
    }
    
    print("3. 开始训练...")
    print("-" * 80)
    
    update_counter = 0
    best_hunter_reward = -float('inf')
    
    for episode in range(1, num_episodes + 1):
        h_obs, t_obs = env.reset()
        episode_rewards_hunters = np.zeros(env.num_hunters)
        episode_rewards_targets = np.zeros(env.num_targets)
        episode_alignment_rewards = []
        
        # 记录初始分配
        initial_assignments = [id(h.assigned_target) if h.assigned_target else None 
                              for h in env.hunters]
        assignment_changes = 0
        
        done = False
        current_step = 1
        capture_success = False
        
        while (not done) and (current_step <= max_steps):
            # 选择动作
            actions_hunters = [hunter.select_action(h_obs[i]) 
                              for i, hunter in enumerate(hunters)]
            actions_targets = [target.select_action(t_obs[i]) 
                              for i, target in enumerate(targets)]
            actions = actions_hunters + actions_targets
            
            # 执行动作
            h_next_obs, t_next_obs, rewards, dones = env.step(actions)
            
            # 计算对齐奖励
            for i, target in enumerate(env.targets):
                v_actual = target.velocity[:2]
                v_ref = target.reference_velocity
                
                if np.linalg.norm(v_actual) > 1e-6 and np.linalg.norm(v_ref) > 1e-6:
                    alignment = np.dot(v_actual, v_ref) / (
                        np.linalg.norm(v_actual) * np.linalg.norm(v_ref)
                    )
                    episode_alignment_rewards.append(alignment)
            
            # 检查分配变化
            current_assignments = [id(h.assigned_target) if h.assigned_target else None 
                                  for h in env.hunters]
            if current_assignments != initial_assignments:
                assignment_changes += 1
                initial_assignments = current_assignments
            
            # 存储transitions
            rewards_hunters = rewards[:env.num_hunters]
            rewards_targets = rewards[env.num_hunters:]
            dones_hunters = dones[:env.num_hunters]
            dones_targets = dones[env.num_hunters:]
            
            for i in range(env.num_hunters):
                hunters_buffer.store_transition(
                    h_obs[i], actions_hunters[i], rewards_hunters[i], 
                    h_next_obs[i], dones_hunters[i]
                )
            
            for i in range(env.num_targets):
                targets_buffer.store_transition(
                    t_obs[i], actions_targets[i], rewards_targets[i], 
                    t_next_obs[i], dones_targets[i]
                )
            
            episode_rewards_hunters += rewards_hunters
            episode_rewards_targets += rewards_targets
            
            h_obs = h_next_obs
            t_obs = t_next_obs
            done = all(dones)
            
            if done:
                capture_success = True
            
            current_step += 1
            
            # 更新网络
            update_counter += 1
            if update_counter % 10 == 0:
                if hunters_buffer.size() >= 1024:
                    for _ in range(20):
                        batch = hunters_buffer.sample(256)
                        for hunter in hunters:
                            hunter.update(batch)
                
                if targets_buffer.size() >= 1024:
                    for _ in range(20):
                        batch = targets_buffer.sample(256)
                        for target in targets:
                            target.update(batch)
        
        # 记录统计
        total_reward_hunters = episode_rewards_hunters.sum()
        total_reward_targets = episode_rewards_targets.sum()
        avg_alignment = np.mean(episode_alignment_rewards) if episode_alignment_rewards else 0
        
        rewards_history['hunters'].append(total_reward_hunters)
        rewards_history['targets'].append(total_reward_targets)
        rewards_history['alignment'].append(avg_alignment)
        rewards_history['density_changes'].append(assignment_changes)
        rewards_history['capture_success'].append(1 if capture_success else 0)
        
        # 保存到CSV
        with open(rewards_csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([episode, total_reward_hunters, total_reward_targets, 
                           avg_alignment, assignment_changes, 1 if capture_success else 0])
        
        # 打印进度
        if episode % 50 == 0:
            recent_hunter_reward = np.mean(rewards_history['hunters'][-50:])
            recent_alignment = np.mean(rewards_history['alignment'][-50:])
            recent_capture = np.mean(rewards_history['capture_success'][-50:])
            
            print(f"Episode {episode}/{num_episodes}:")
            print(f"  最近50轮Hunter平均奖励: {recent_hunter_reward:.2f}")
            print(f"  最近50轮Target平均对齐度: {recent_alignment:.4f}")
            print(f"  最近50轮捕获成功率: {recent_capture:.2%}")
            print(f"  当前轮分配变化: {assignment_changes}次")
        
        # 保存最佳模型
        if total_reward_hunters > best_hunter_reward:
            best_hunter_reward = total_reward_hunters
            save_dir = os.path.join(os.getcwd(), "model", 
                                   f"{timestamp}_full_training_best")
            os.makedirs(save_dir, exist_ok=True)
            
            for i, hunter in enumerate(hunters):
                hunter.save_model(save_dir, agent_id=i, agent_type='hunter')
            
            for i, target in enumerate(targets):
                target.save_model(save_dir, agent_id=i, agent_type='target')
    
    print("-" * 80)
    print()
    
    # 保存最终模型
    final_save_dir = os.path.join(os.getcwd(), "model", 
                                 f"{timestamp}_full_training_final")
    os.makedirs(final_save_dir, exist_ok=True)
    
    for i, hunter in enumerate(hunters):
        hunter.save_model(final_save_dir, agent_id=i, agent_type='hunter')
    
    for i, target in enumerate(targets):
        target.save_model(final_save_dir, agent_id=i, agent_type='target')
    
    print(f"4. 模型已保存:")
    print(f"   - 最佳模型: {timestamp}_full_training_best")
    print(f"   - 最终模型: {timestamp}_full_training_final")
    print()
    
    # 生成训练曲线
    print("5. 生成训练分析...")
    generate_training_analysis(rewards_history, data_dir)
    
    env.close()
    return rewards_history, data_dir

def generate_training_analysis(rewards_history, data_dir):
    """生成训练分析图表"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    episodes = range(1, len(rewards_history['hunters']) + 1)
    
    # 1. Hunter奖励
    axes[0, 0].plot(episodes, rewards_history['hunters'], alpha=0.3, color='blue')
    # 移动平均
    window = 50
    if len(rewards_history['hunters']) >= window:
        moving_avg = np.convolve(rewards_history['hunters'], 
                                np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window, len(rewards_history['hunters'])+1), 
                       moving_avg, color='blue', linewidth=2, label='50-episode MA')
    axes[0, 0].set_title('Hunter Total Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Target奖励
    axes[0, 1].plot(episodes, rewards_history['targets'], alpha=0.3, color='red')
    if len(rewards_history['targets']) >= window:
        moving_avg = np.convolve(rewards_history['targets'], 
                                np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window, len(rewards_history['targets'])+1), 
                       moving_avg, color='red', linewidth=2, label='50-episode MA')
    axes[0, 1].set_title('Target Total Reward')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Reward')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. 对齐度
    axes[1, 0].plot(episodes, rewards_history['alignment'], alpha=0.3, color='green')
    if len(rewards_history['alignment']) >= window:
        moving_avg = np.convolve(rewards_history['alignment'], 
                                np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window, len(rewards_history['alignment'])+1), 
                       moving_avg, color='green', linewidth=2, label='50-episode MA')
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Target Velocity Alignment (Cosine Similarity)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Alignment')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 4. 分配变化
    axes[1, 1].plot(episodes, rewards_history['density_changes'], 
                   alpha=0.3, color='orange')
    if len(rewards_history['density_changes']) >= window:
        moving_avg = np.convolve(rewards_history['density_changes'], 
                                np.ones(window)/window, mode='valid')
        axes[1, 1].plot(range(window, len(rewards_history['density_changes'])+1), 
                       moving_avg, color='orange', linewidth=2, label='50-episode MA')
    axes[1, 1].set_title('Density Field Assignment Changes')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Changes per Episode')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 5. 捕获成功率
    axes[2, 0].plot(episodes, rewards_history['capture_success'], 
                   alpha=0.3, color='purple')
    if len(rewards_history['capture_success']) >= window:
        moving_avg = np.convolve(rewards_history['capture_success'], 
                                np.ones(window)/window, mode='valid')
        axes[2, 0].plot(range(window, len(rewards_history['capture_success'])+1), 
                       moving_avg, color='purple', linewidth=2, label='50-episode MA')
    axes[2, 0].set_title('Capture Success Rate')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Success (1=Yes, 0=No)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # 6. 学习曲线对比
    # 分段统计
    segments = 5
    segment_size = len(rewards_history['hunters']) // segments
    segment_means_h = []
    segment_means_t = []
    segment_labels = []
    
    for i in range(segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < segments - 1 else len(rewards_history['hunters'])
        segment_means_h.append(np.mean(rewards_history['hunters'][start:end]))
        segment_means_t.append(np.mean(rewards_history['targets'][start:end]))
        segment_labels.append(f"{start+1}-{end}")
    
    x = np.arange(len(segment_labels))
    width = 0.35
    axes[2, 1].bar(x - width/2, segment_means_h, width, label='Hunters', color='blue', alpha=0.7)
    axes[2, 1].bar(x + width/2, segment_means_t, width, label='Targets', color='red', alpha=0.7)
    axes[2, 1].set_title('Average Rewards by Training Phase')
    axes[2, 1].set_xlabel('Episode Range')
    axes[2, 1].set_ylabel('Average Reward')
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(segment_labels, rotation=45)
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'training_analysis.png'), dpi=150)
    print(f"   ✓ 训练分析图已保存")

if __name__ == "__main__":
    try:
        print("开始完整训练...")
        print("预计时间: 约30-40分钟")
        print()
        
        rewards_history, data_dir = full_training()
        
        print()
        print("=" * 80)
        print("训练完成！")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
