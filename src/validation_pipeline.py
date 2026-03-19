"""
验证流水线 (Validation Pipeline)

自动化模型验证和结果生成系统，支持批量测试训练好的模型并生成标准化报告。

功能:
- 支持YAML和JSON格式的配置文件
- 自动加载训练模型
- 执行指定轮数的验证回合
- 按帧间隔保存验证图片
- 生成验证视频和日志文件
- 记录模型配置和随机种子信息
"""

import os
import json
import yaml
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from MultiTargetEnv import MultiTarEnv, set_global_seeds
from MATD3 import MATD3Agent


def setup_chinese_font():
    """
    配置matplotlib中文字体
    
    使用宋体(SimSun)显示中文，解决中文乱码问题。
    同时修复负号显示问题。
    """
    rcParams['font.sans-serif'] = ['SimSun', 'DejaVu Sans']  # 宋体，备用DejaVu Sans
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class ValidationPipeline:
    """
    自动验证流水线类
    
    负责加载配置、执行模型验证、保存结果和生成报告。
    """
    
    def __init__(self, config_path: str):
        """
        初始化验证流水线
        
        参数:
            config_path: 配置文件路径（YAML或JSON格式）
        
        异常:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误或缺少必需字段
        """
        self.config = self.load_config(config_path)
        self.output_dir = self._create_output_dir()

        # Initialize environment and agents (will be set in load_model)
        self.env = None
        self.hunters = []
        self.targets = []

        # Metrics storage
        self.episode_metrics = []

        # Headless mode: 不保存中间图片，直接从内存帧生成视频
        self.headless = self.config.get('headless', False)
        self._episode_frames = []  # 内存帧缓存

        # Setup Chinese font for matplotlib
        setup_chinese_font()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载并验证配置文件
        
        支持YAML (.yaml, .yml) 和 JSON (.json) 格式。
        自动验证必需字段并提供默认值。
        
        参数:
            config_path: 配置文件路径
        
        返回:
            dict: 解析后的配置字典
        
        异常:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误或缺少必需字段
        """
        # 检查文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 根据文件扩展名选择解析器
        try:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path}。仅支持 .yaml, .yml, .json")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML配置文件解析失败: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON配置文件解析失败: {str(e)}")
        except Exception as e:
            raise ValueError(f"配置文件读取失败: {str(e)}")
        
        # 验证配置完整性
        self._validate_config(config)
        
        # 提供默认值
        self._set_default_values(config)
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证配置文件包含所有必需字段
        
        必需字段:
        - model.path: 模型路径
        - validation.num_episodes: 验证回合数
        
        参数:
            config: 配置字典
        
        异常:
            ValueError: 缺少必需字段
        """
        required_fields = [
            ('model', 'path'),
            ('validation', 'num_episodes')
        ]
        
        for field_path in required_fields:
            value = self._get_nested_value(config, field_path)
            if value is None:
                field_name = '.'.join(field_path)
                raise ValueError(f"配置文件缺少必需字段: {field_name}")
    
    def _get_nested_value(self, config: Dict[str, Any], field_path: tuple) -> Optional[Any]:
        """
        获取嵌套字典中的值
        
        参数:
            config: 配置字典
            field_path: 字段路径元组，例如 ('model', 'path')
        
        返回:
            字段值，如果不存在则返回None
        """
        current = config
        for key in field_path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current
    
    def _set_default_values(self, config: Dict[str, Any]) -> None:
        """
        为配置设置默认值
        
        默认值:
        - validation.max_steps: 150
        - validation.save_frame_interval: 5
        - random_seed: 42
        - output.save_images: True
        - output.save_video: True
        - output.save_logs: True
        - output.video_fps: 10
        - environment.num_hunters: 6
        - environment.num_targets: 2
        - environment.num_obstacles: 5
        - environment.env_length: 2.0
        
        参数:
            config: 配置字典（会被就地修改）
        """
        # 确保嵌套字典存在
        if 'validation' not in config:
            config['validation'] = {}
        if 'output' not in config:
            config['output'] = {}
        if 'environment' not in config:
            config['environment'] = {}
        
        # 设置默认值
        config['validation'].setdefault('max_steps', 150)
        config['validation'].setdefault('save_frame_interval', 5)
        config.setdefault('random_seed', 42)
        config['output'].setdefault('save_images', True)
        config['output'].setdefault('save_video', True)
        config['output'].setdefault('save_logs', True)
        config['output'].setdefault('video_fps', 10)
        
        # Environment defaults
        config['environment'].setdefault('num_hunters', 6)
        config['environment'].setdefault('num_targets', 2)
        config['environment'].setdefault('num_obstacles', 5)
        config['environment'].setdefault('env_length', 2.0)
    
    def _create_output_dir(self) -> str:
        """
        创建输出目录结构
        
        目录结构:
        output/
        └── {timestamp}_validation/
            ├── images/      # 验证过程图片
            ├── videos/      # 验证视频
            ├── logs/        # 日志文件
            └── config/      # 配置文件备份
        
        返回:
            str: 输出目录的完整路径
        """
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建主输出目录
        output_base = os.path.join('output', f'{timestamp}_validation')
        
        # 创建子目录
        subdirs = ['images', 'videos', 'logs', 'config']
        for subdir in subdirs:
            subdir_path = os.path.join(output_base, subdir)
            os.makedirs(subdir_path, exist_ok=True)
        
        return output_base
    
    def get_output_subdir(self, subdir_name: str) -> str:
        """
        获取输出子目录的完整路径
        
        参数:
            subdir_name: 子目录名称 ('images', 'videos', 'logs', 'config')
        
        返回:
            str: 子目录的完整路径
        """
        return os.path.join(self.output_dir, subdir_name)
    
    def save_config_backup(self) -> None:
        """
        保存配置文件的备份到输出目录
        
        将当前使用的配置保存为JSON格式，便于追溯验证参数。
        """
        config_backup_path = os.path.join(
            self.get_output_subdir('config'),
            'validation_config.json'
        )
        
        with open(config_backup_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def run_validation(self):
        """
        执行完整的验证流程
        
        流程:
        1. 保存配置备份
        2. 设置随机种子
        3. 加载模型
        4. 运行验证回合
        5. 记录指标
        6. 生成视频
        7. 生成报告
        """
        # 保存配置备份
        self.save_config_backup()
        
        # 设置随机种子确保可重复性
        random_seed = self.config.get('random_seed', 42)
        set_global_seeds(random_seed)
        print(f"随机种子设置为: {random_seed}")
        
        # 加载模型
        print("正在加载模型...")
        self.load_model()
        print("模型加载完成")
        
        # 运行验证回合
        num_episodes = self.config['validation']['num_episodes']
        max_steps = self.config['validation']['max_steps']
        
        print(f"\n开始验证: {num_episodes} 回合, 每回合最多 {max_steps} 步")
        print(f"输出目录: {self.output_dir}")
        
        for episode in range(num_episodes):
            print(f"\n--- 回合 {episode + 1}/{num_episodes} ---")
            metrics = self._run_single_episode(episode, max_steps)
            self.episode_metrics.append(metrics)
            
            # 打印回合摘要
            print(f"  步数: {metrics['total_steps']}")
            print(f"  捕获成功: {metrics['capture_success']}")
            print(f"  平均Hunter奖励: {metrics['avg_hunter_reward']:.4f}")
            print(f"  平均Target奖励: {metrics['avg_target_reward']:.4f}")
            
            # 生成视频（如果启用）
            if self.config['output'].get('save_video', True):
                print(f"  正在生成视频...")
                if self.headless:
                    self._generate_video_from_frames(episode)
                else:
                    self.generate_video(episode)
        
        # 生成最终报告
        print("\n生成验证报告...")
        self._save_metrics_log()
        self.generate_report()
        print(f"\n验证完成! 结果保存在: {self.output_dir}")
    
    def load_model(self):
        """
        加载hunter和target的MATD3模型
        
        从配置文件指定的路径加载训练好的模型，并初始化环境和智能体。
        自动检测模型的观测维度以兼容不同版本的模型。
        
        异常:
            FileNotFoundError: 模型文件不存在
            ValueError: 模型加载失败
        """
        model_path = self.config['model']['path']
        
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 获取环境配置
        env_config = self.config['environment']
        num_hunters = env_config['num_hunters']
        num_targets = env_config['num_targets']
        num_obstacles = env_config['num_obstacles']
        env_length = env_config['env_length']
        
        # 检测模型的观测维度
        h_actor_dim, t_actor_dim = self._detect_model_dimensions(model_path)
        print(f"检测到模型维度: Hunter={h_actor_dim}, Target={t_actor_dim}")
        
        action_dim = 2
        
        # 初始化环境
        self.env = MultiTarEnv(
            length=env_length,
            num_obstacle=num_obstacles,
            num_hunters=num_hunters,
            num_targets=num_targets,
            h_actor_dim=h_actor_dim,
            t_actor_dim=t_actor_dim,
            action_dim=action_dim,
            visualize_lasers=False  # Don't visualize during validation
        )
        
        # 初始化hunter agents
        self.hunters = []
        for i in range(num_hunters):
            agent = MATD3Agent(
                obs_dim=h_actor_dim,
                action_dim=action_dim,
                lr=1e-3,
                gamma=0.95,
                tau=0.01,
                noise_std=0.1,  # Standard noise for exploration
                a_max=self.env.a_max,
                device='cpu'
            )
            # 加载模型
            try:
                agent.load_model(model_path, agent_id=i, agent_type='hunter')
                print(f"  已加载 hunter_{i} 模型")
            except Exception as e:
                raise ValueError(f"加载 hunter_{i} 模型失败: {str(e)}")
            
            self.hunters.append(agent)
        
        # 初始化target agents
        self.targets = []
        for i in range(num_targets):
            agent = MATD3Agent(
                obs_dim=t_actor_dim,
                action_dim=action_dim,
                lr=1e-3,
                gamma=0.95,
                tau=0.01,
                noise_std=0.1,  # Standard noise for exploration
                a_max=self.env.a_max,
                device='cpu'
            )
            # 加载模型
            try:
                agent.load_model(model_path, agent_id=i, agent_type='target')
                print(f"  已加载 target_{i} 模型")
            except Exception as e:
                raise ValueError(f"加载 target_{i} 模型失败: {str(e)}")
            
            self.targets.append(agent)
    
    def _detect_model_dimensions(self, model_path: str) -> tuple:
        """
        检测模型的观测维度
        
        通过加载模型文件并检查第一层的权重形状来确定观测维度。
        
        参数:
            model_path: 模型目录路径
        
        返回:
            tuple: (hunter_obs_dim, target_obs_dim)
        """
        import torch
        
        # 检测hunter维度
        hunter_actor_path = os.path.join(model_path, 'hunter_0', 'actor.pth')
        if os.path.exists(hunter_actor_path):
            hunter_state = torch.load(hunter_actor_path, map_location='cpu')
            # fc1.weight shape is [128, obs_dim]
            h_obs_dim = hunter_state['fc1.weight'].shape[1]
        else:
            # 默认值
            h_obs_dim = 32
            print(f"警告: 未找到hunter模型文件，使用默认维度 {h_obs_dim}")
        
        # 检测target维度
        target_actor_path = os.path.join(model_path, 'target_0', 'actor.pth')
        if os.path.exists(target_actor_path):
            target_state = torch.load(target_actor_path, map_location='cpu')
            # fc1.weight shape is [128, obs_dim]
            t_obs_dim = target_state['fc1.weight'].shape[1]
        else:
            # 默认值
            t_obs_dim = 31
            print(f"警告: 未找到target模型文件，使用默认维度 {t_obs_dim}")
        
        return h_obs_dim, t_obs_dim
    
    def _run_single_episode(self, episode: int, max_steps: int) -> Dict[str, Any]:
        """
        运行单个验证回合
        
        参数:
            episode: 回合编号
            max_steps: 最大步数
        
        返回:
            dict: 回合指标
        """
        # 重置环境
        h_obs, t_obs = self.env.reset()
        
        # 检测是否需要调整target观测维度
        # 环境总是生成包含reference_velocity的观测(33维)
        # 但旧模型可能期望31维(不包含reference_velocity)
        target_obs_dim = self.targets[0].obs_dim if self.targets else 31
        env_target_obs_dim = len(t_obs[0]) if t_obs else 33
        trim_target_obs = (target_obs_dim < env_target_obs_dim)
        
        if trim_target_obs:
            # 移除最后2维(reference_velocity)
            t_obs = [obs[:target_obs_dim] for obs in t_obs]
        
        # 初始化指标
        total_steps = 0
        capture_success = False
        hunter_rewards = []
        target_rewards = []
        
        # 获取保存帧的间隔
        save_frame_interval = self.config['validation'].get('save_frame_interval', 5)
        save_images = self.config['output'].get('save_images', True)
        self._episode_frames = []  # 清空内存帧缓存

        # 运行回合
        for step in range(max_steps):
            # 选择动作 (使用确定性策略进行验证，不添加噪声)
            h_actions = []
            for i, hunter in enumerate(self.hunters):
                action = hunter.select_action(h_obs[i], noise=False)
                h_actions.append(action)

            t_actions = []
            for i, target in enumerate(self.targets):
                action = target.select_action(t_obs[i], noise=False)
                t_actions.append(action)

            # 合并动作: hunters first, then targets
            all_actions = h_actions + t_actions

            # 执行动作
            h_obs_next, t_obs_next, rewards, dones, _reward_info = self.env.step(all_actions)

            # 保存帧
            if step % save_frame_interval == 0:
                if self.headless:
                    # 无头模式：渲染到内存帧
                    frame = self._render_frame_to_array(self.env, episode, step)
                    if frame is not None:
                        self._episode_frames.append(frame)
                elif save_images:
                    self.save_frame(self.env, episode, step)
            
            # 分离hunter和target的奖励
            h_rewards = rewards[:self.env.num_hunters]
            t_rewards = rewards[self.env.num_hunters:]
            
            # 检查done标志
            done = any(dones)
            
            # 调整target观测维度(如果需要)
            if trim_target_obs:
                t_obs_next = [obs[:target_obs_dim] for obs in t_obs_next]
            
            # 记录奖励
            hunter_rewards.append(np.mean(h_rewards))
            target_rewards.append(np.mean(t_rewards))
            
            # 更新观测
            h_obs = h_obs_next
            t_obs = t_obs_next
            
            total_steps = step + 1
            
            # 检查是否完成
            if done:
                capture_success = True
                break
        
        # 保存最后一帧
        if self.headless:
            frame = self._render_frame_to_array(self.env, episode, total_steps)
            if frame is not None:
                self._episode_frames.append(frame)
        elif save_images:
            self.save_frame(self.env, episode, total_steps)
        
        # 计算指标
        metrics = {
            'episode': episode,
            'total_steps': total_steps,
            'capture_success': capture_success,
            'avg_hunter_reward': np.mean(hunter_rewards) if hunter_rewards else 0.0,
            'avg_target_reward': np.mean(target_rewards) if target_rewards else 0.0,
            'total_hunter_reward': np.sum(hunter_rewards) if hunter_rewards else 0.0,
            'total_target_reward': np.sum(target_rewards) if target_rewards else 0.0,
            'hunter_rewards': hunter_rewards,
            'target_rewards': target_rewards
        }
        
        return metrics
    
    def _save_metrics_log(self):
        """
        保存验证指标日志到文件
        
        生成包含每轮验证关键指标的日志文件。
        """
        log_path = os.path.join(self.get_output_subdir('logs'), 'validation_metrics.txt')
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("验证指标报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 写入配置信息
            f.write("配置信息:\n")
            f.write(f"  模型路径: {self.config['model']['path']}\n")
            f.write(f"  随机种子: {self.config['random_seed']}\n")
            f.write(f"  验证回合数: {self.config['validation']['num_episodes']}\n")
            f.write(f"  最大步数: {self.config['validation']['max_steps']}\n")
            f.write(f"  环境配置: {self.config['environment']}\n")
            f.write("\n" + "-" * 80 + "\n\n")
            
            # 写入每回合指标
            f.write("每回合详细指标:\n\n")
            for metrics in self.episode_metrics:
                f.write(f"回合 {metrics['episode'] + 1}:\n")
                f.write(f"  总步数: {metrics['total_steps']}\n")
                f.write(f"  捕获成功: {'是' if metrics['capture_success'] else '否'}\n")
                f.write(f"  平均Hunter奖励: {metrics['avg_hunter_reward']:.4f}\n")
                f.write(f"  平均Target奖励: {metrics['avg_target_reward']:.4f}\n")
                f.write(f"  总Hunter奖励: {metrics['total_hunter_reward']:.4f}\n")
                f.write(f"  总Target奖励: {metrics['total_target_reward']:.4f}\n")
                f.write("\n")
            
            # 写入汇总统计
            f.write("-" * 80 + "\n\n")
            f.write("汇总统计:\n\n")
            
            success_count = sum(1 for m in self.episode_metrics if m['capture_success'])
            success_rate = success_count / len(self.episode_metrics) if self.episode_metrics else 0
            
            avg_steps = np.mean([m['total_steps'] for m in self.episode_metrics])
            avg_hunter_reward = np.mean([m['avg_hunter_reward'] for m in self.episode_metrics])
            avg_target_reward = np.mean([m['avg_target_reward'] for m in self.episode_metrics])
            
            f.write(f"  捕获成功率: {success_rate:.2%} ({success_count}/{len(self.episode_metrics)})\n")
            f.write(f"  平均步数: {avg_steps:.2f}\n")
            f.write(f"  平均Hunter奖励: {avg_hunter_reward:.4f}\n")
            f.write(f"  平均Target奖励: {avg_target_reward:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"指标日志已保存到: {log_path}")
    
    def save_frame(self, env, episode: int, step: int):
        """
        保存单帧图片
        
        使用环境的render方法生成当前状态的可视化图片，并保存到images目录。
        图片文件名格式: episode_{episode}_step_{step}.png
        
        参数:
            env: 环境实例
            episode: 当前回合编号
            step: 当前步数
        """
        # 创建回合专属的图片目录
        episode_image_dir = os.path.join(
            self.get_output_subdir('images'),
            f'episode_{episode}'
        )
        os.makedirs(episode_image_dir, exist_ok=True)
        
        # 生成文件名
        filename = f'step_{step:04d}.png'
        filepath = os.path.join(episode_image_dir, filename)
        
        # 渲染环境并保存
        # 假设环境有render方法返回matplotlib figure
        # 如果环境没有返回figure，我们需要手动创建
        try:
            # 尝试使用环境的render方法
            if hasattr(env, 'render') and callable(env.render):
                env.render()
                
                # 保存当前figure
                if hasattr(env, 'fig') and env.fig is not None:
                    env.fig.savefig(filepath, dpi=100, bbox_inches='tight')
                else:
                    # 如果环境没有fig属性，使用当前活动的figure
                    plt.savefig(filepath, dpi=100, bbox_inches='tight')
            else:
                # 如果环境没有render方法，创建简单的可视化
                self._create_simple_visualization(env, filepath)
        except Exception as e:
            print(f"警告: 保存帧失败 (episode={episode}, step={step}): {str(e)}")
    
    def _create_simple_visualization(self, env, filepath: str):
        """
        创建简单的环境可视化
        
        当环境没有render方法时使用的备用可视化方案。
        
        参数:
            env: 环境实例
            filepath: 保存路径
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制hunters
        if hasattr(env, 'hunters'):
            for hunter in env.hunters:
                pos = hunter.position
                ax.scatter(pos[0], pos[1], pos[2], c='red', marker='o', s=100, label='Hunter')
        
        # 绘制targets
        if hasattr(env, 'targets'):
            for target in env.targets:
                pos = target.position
                ax.scatter(pos[0], pos[1], pos[2], c='green', marker='^', s=100, label='Target')
        
        # 绘制obstacles
        if hasattr(env, 'obstacles'):
            for obstacle in env.obstacles:
                pos = obstacle.position
                ax.scatter(pos[0], pos[1], pos[2], c='gray', marker='s', s=200, alpha=0.5, label='Obstacle')
        
        # 设置坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('环境状态')
        
        # 保存并关闭
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def _render_frame_to_array(self, env, episode: int, step: int):
        """无头模式：渲染当前环境状态为内存图像数组"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.set_xlim(-0.05, env.length + 0.05)
            ax.set_ylim(-0.05, env.length + 0.05)
            ax.set_aspect('equal')
            ax.set_title(f'Episode {episode+1} | Step {step}', fontsize=14)

            # 边界
            ax.plot([0, env.length, env.length, 0, 0],
                    [0, 0, env.length, env.length, 0], 'k-', lw=2)

            # 障碍物
            for obs in env.obstacles:
                cx, cy, _, r, _ = obs._return_obs_info()
                circle = plt.Circle((cx, cy), r, color='gray', alpha=0.5)
                ax.add_patch(circle)

            # 逃逸区域
            if hasattr(env, 'escape_zone_center'):
                esc = plt.Circle(env.escape_zone_center, env.escape_zone_radius,
                                 color='green', alpha=0.2, linestyle='--', linewidth=2, fill=True)
                ax.add_patch(esc)
                ax.annotate('EXIT', xy=env.escape_zone_center, ha='center', va='center',
                            fontsize=10, color='green', fontweight='bold')

            # 猎手
            h_colors = ['#e74c3c', '#c0392b', '#e67e22', '#d35400', '#f39c12', '#e84393']
            for i, hunter in enumerate(env.hunters):
                c = h_colors[i % len(h_colors)]
                # 轨迹
                if hasattr(hunter, 'history_pos') and len(hunter.history_pos) > 1:
                    traj = np.array(hunter.history_pos)
                    ax.plot(traj[:, 0], traj[:, 1], '-', color=c, alpha=0.3, lw=1)
                ax.plot(hunter.position[0], hunter.position[1], 'o', color=c,
                        markersize=8, markeredgecolor='black', markeredgewidth=0.5)

            # 目标
            t_colors = ['#2ecc71', '#27ae60']
            for i, target in enumerate(env.targets):
                c = t_colors[i % len(t_colors)]
                if hasattr(target, 'history_pos') and len(target.history_pos) > 1:
                    traj = np.array(target.history_pos)
                    ax.plot(traj[:, 0], traj[:, 1], '-', color=c, alpha=0.3, lw=1)
                ax.plot(target.position[0], target.position[1], '^', color=c,
                        markersize=10, markeredgecolor='black', markeredgewidth=0.5)

            # 转为图像数组
            fig.canvas.draw()
            w, h_px = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h_px, w, 3)
            img = img.copy()  # 脱离buffer引用
            plt.close(fig)
            return img
        except Exception as e:
            print(f"警告: 渲染帧失败 (episode={episode}, step={step}): {str(e)}")
            return None

    def _generate_video_from_frames(self, episode: int):
        """无头模式：从内存帧直接生成视频，不经过磁盘图片"""
        if not self._episode_frames:
            print(f"  警告: 无帧可用于生成视频")
            return

        video_path = os.path.join(self.get_output_subdir('videos'), f'episode_{episode}.mp4')
        h_px, w, _ = self._episode_frames[0].shape
        fps = self.config['output'].get('video_fps', 10)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h_px))

        for frame in self._episode_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        self._episode_frames = []
        print(f"  视频已生成: {video_path}")
    
    def generate_video(self, episode: int):
        """
        从图片序列生成视频
        
        使用OpenCV将指定回合的所有图片帧合成为视频文件。
        视频文件名格式: episode_{episode}.mp4
        
        参数:
            episode: 回合编号
        """
        # 获取图片目录
        episode_image_dir = os.path.join(
            self.get_output_subdir('images'),
            f'episode_{episode}'
        )
        
        # 检查目录是否存在
        if not os.path.exists(episode_image_dir):
            print(f"警告: 图片目录不存在: {episode_image_dir}")
            return
        
        # 获取所有图片文件
        image_files = sorted([
            f for f in os.listdir(episode_image_dir)
            if f.endswith('.png')
        ])
        
        if len(image_files) == 0:
            print(f"警告: 没有找到图片文件: {episode_image_dir}")
            return
        
        # 读取第一张图片以获取尺寸
        first_image_path = os.path.join(episode_image_dir, image_files[0])
        first_image = cv2.imread(first_image_path)
        if first_image is None:
            print(f"警告: 无法读取图片: {first_image_path}")
            return
        
        height, width, _ = first_image.shape
        
        # 创建视频写入器
        video_filename = f'episode_{episode}.mp4'
        video_path = os.path.join(self.get_output_subdir('videos'), video_filename)
        
        fps = self.config['output'].get('video_fps', 10)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # 写入所有帧
        for image_file in image_files:
            image_path = os.path.join(episode_image_dir, image_file)
            frame = cv2.imread(image_path)
            if frame is not None:
                video_writer.write(frame)
        
        # 释放资源
        video_writer.release()
        print(f"  视频已生成: {video_path}")
    
    def generate_report(self):
        """
        生成验证报告
        
        生成包含以下内容的完整验证报告:
        1. 配置信息备份 (已在save_config_backup中完成)
        2. 验证指标日志 (已在_save_metrics_log中完成)
        3. 汇总报告 (markdown格式)
        
        此方法主要负责生成markdown格式的汇总报告。
        """
        report_path = os.path.join(self.get_output_subdir('logs'), 'validation_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 验证报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 配置信息
            f.write("## 配置信息\n\n")
            f.write(f"- **模型路径**: `{self.config['model']['path']}`\n")
            f.write(f"- **随机种子**: {self.config['random_seed']}\n")
            f.write(f"- **验证回合数**: {self.config['validation']['num_episodes']}\n")
            f.write(f"- **最大步数**: {self.config['validation']['max_steps']}\n")
            f.write(f"- **帧保存间隔**: {self.config['validation']['save_frame_interval']}\n")
            f.write(f"- **视频FPS**: {self.config['output']['video_fps']}\n")
            f.write("\n### 环境配置\n\n")
            env_config = self.config['environment']
            f.write(f"- **Hunter数量**: {env_config['num_hunters']}\n")
            f.write(f"- **Target数量**: {env_config['num_targets']}\n")
            f.write(f"- **障碍物数量**: {env_config['num_obstacles']}\n")
            f.write(f"- **环境大小**: {env_config['env_length']}\n")
            f.write("\n")
            
            # 汇总统计
            f.write("## 汇总统计\n\n")
            
            if self.episode_metrics:
                success_count = sum(1 for m in self.episode_metrics if m['capture_success'])
                success_rate = success_count / len(self.episode_metrics)
                
                avg_steps = np.mean([m['total_steps'] for m in self.episode_metrics])
                std_steps = np.std([m['total_steps'] for m in self.episode_metrics])
                
                avg_hunter_reward = np.mean([m['avg_hunter_reward'] for m in self.episode_metrics])
                avg_target_reward = np.mean([m['avg_target_reward'] for m in self.episode_metrics])
                
                f.write(f"- **捕获成功率**: {success_rate:.2%} ({success_count}/{len(self.episode_metrics)})\n")
                f.write(f"- **平均步数**: {avg_steps:.2f} ± {std_steps:.2f}\n")
                f.write(f"- **平均Hunter奖励**: {avg_hunter_reward:.4f}\n")
                f.write(f"- **平均Target奖励**: {avg_target_reward:.4f}\n")
                f.write("\n")
            
            # 每回合详情
            f.write("## 每回合详情\n\n")
            f.write("| 回合 | 步数 | 捕获成功 | Hunter奖励 | Target奖励 |\n")
            f.write("|------|------|----------|------------|------------|\n")
            
            for metrics in self.episode_metrics:
                episode_num = metrics['episode'] + 1
                steps = metrics['total_steps']
                success = '✓' if metrics['capture_success'] else '✗'
                h_reward = metrics['avg_hunter_reward']
                t_reward = metrics['avg_target_reward']
                
                f.write(f"| {episode_num} | {steps} | {success} | {h_reward:.4f} | {t_reward:.4f} |\n")
            
            f.write("\n")
            
            # 输出文件说明
            f.write("## 输出文件\n\n")
            f.write("验证结果保存在以下目录:\n\n")
            f.write(f"```\n{self.output_dir}/\n")
            f.write("├── images/          # 验证过程图片\n")
            f.write("│   ├── episode_0/\n")
            f.write("│   ├── episode_1/\n")
            f.write("│   └── ...\n")
            f.write("├── videos/          # 验证视频\n")
            f.write("│   ├── episode_0.mp4\n")
            f.write("│   ├── episode_1.mp4\n")
            f.write("│   └── ...\n")
            f.write("├── logs/            # 日志文件\n")
            f.write("│   ├── validation_metrics.txt\n")
            f.write("│   └── validation_report.md\n")
            f.write("└── config/          # 配置备份\n")
            f.write("    └── validation_config.json\n")
            f.write("```\n")
        
        print(f"验证报告已生成: {report_path}")


if __name__ == '__main__':
    # 示例用法
    print("ValidationPipeline 类已创建")
    print("使用示例:")
    print("  pipeline = ValidationPipeline('config.yaml')")
    print("  pipeline.run_validation()")
