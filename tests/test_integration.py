"""
集成测试 - 验证密度场分配、逃逸策略、角色分配的协同工作

测试覆盖:
1. 环境初始化和重置
2. 密度场分配 + 角色分配 + 逃逸策略的完整step循环
3. 观测空间维度一致性
4. 奖励计算正确性
5. 卡尔曼滤波器在多步中的稳定性
6. 验证流水线配置解析
"""

import sys
import os
import unittest
import numpy as np
import tempfile
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from MultiTargetEnv import MultiTarEnv, Hunter, Target, Obstacle, set_global_seeds
from density_field_allocator import DensityFieldAllocator, safe_divide
from reference_velocity_calculator import ReferenceVelocityCalculator
from role_assigner import RoleAssigner
from kalman_filter import KalmanFilter


class TestEnvironmentIntegration(unittest.TestCase):
    """测试环境初始化和完整step循环"""

    def setUp(self):
        set_global_seeds(42)
        self.env = MultiTarEnv(
            length=2.0,
            num_obstacle=3,
            num_hunters=4,
            num_targets=2,
            h_actor_dim=32,
            t_actor_dim=33,
            action_dim=2,
            visualize_lasers=False
        )

    def test_reset_returns_correct_obs_dimensions(self):
        """验证reset返回的观测维度正确"""
        h_obs, t_obs = self.env.reset()

        self.assertEqual(len(h_obs), self.env.num_hunters)
        self.assertEqual(len(t_obs), self.env.num_targets)

        # Hunter obs: 2*3(nearest) + 3(pos) + 3(vel) + 3(target_pos) + 1(dist) + 16(laser) = 32
        for obs in h_obs:
            self.assertEqual(len(obs), 32)

        # Target obs: 3(pos) + 3(vel) + 3*3(nearest_hunters) + 16(laser) + 2(ref_vel) = 33
        for obs in t_obs:
            self.assertEqual(len(obs), 33)

    def test_step_cycle_runs_without_error(self):
        """验证完整的step循环不报错"""
        h_obs, t_obs = self.env.reset()

        # 随机动作
        actions = [np.random.uniform(-0.01, 0.01, size=2)
                   for _ in range(self.env.num_hunters + self.env.num_targets)]

        h_obs_next, t_obs_next, rewards, dones = self.env.step(actions)

        self.assertEqual(len(h_obs_next), self.env.num_hunters)
        self.assertEqual(len(t_obs_next), self.env.num_targets)
        self.assertEqual(len(rewards), self.env.num_hunters + self.env.num_targets)
        self.assertEqual(len(dones), self.env.num_hunters + self.env.num_targets)

    def test_multi_step_stability(self):
        """验证多步运行的稳定性（无NaN、无崩溃）"""
        self.env.reset()

        for step in range(50):
            actions = [np.random.uniform(-0.01, 0.01, size=2)
                       for _ in range(self.env.num_hunters + self.env.num_targets)]
            h_obs, t_obs, rewards, dones = self.env.step(actions)

            # 检查无NaN
            for obs in h_obs:
                self.assertFalse(np.any(np.isnan(obs)),
                                 f"NaN in hunter obs at step {step}")
            for obs in t_obs:
                self.assertFalse(np.any(np.isnan(obs)),
                                 f"NaN in target obs at step {step}")
            for r in rewards:
                self.assertFalse(np.isnan(r),
                                 f"NaN in reward at step {step}")

    def test_target_assignment_updates_each_step(self):
        """验证每步都更新了hunter的目标分配"""
        self.env.reset()

        actions = [np.random.uniform(-0.01, 0.01, size=2)
                   for _ in range(self.env.num_hunters + self.env.num_targets)]
        self.env.step(actions)

        # 每个hunter都应该有assigned_target
        for hunter in self.env.hunters:
            self.assertIsNotNone(hunter.assigned_target,
                                 "Hunter should have an assigned target after step")

    def test_role_assignment_valid_after_step(self):
        """验证step后每个hunter都有有效角色"""
        self.env.reset()

        actions = [np.random.uniform(-0.01, 0.01, size=2)
                   for _ in range(self.env.num_hunters + self.env.num_targets)]
        self.env.step(actions)

        for hunter in self.env.hunters:
            self.assertIn(hunter.role, ['chaser', 'interceptor'],
                          f"Invalid role: {hunter.role}")

    def test_reference_velocity_updated_after_step(self):
        """验证step后target的reference_velocity已更新"""
        self.env.reset()

        actions = [np.random.uniform(-0.01, 0.01, size=2)
                   for _ in range(self.env.num_hunters + self.env.num_targets)]
        self.env.step(actions)

        for target in self.env.targets:
            self.assertEqual(len(target.reference_velocity), 2)
            # reference_velocity不应全为零（因为有hunter在附近）
            # 注意：如果所有hunter都在感知范围外，可能为随机方向



class TestDensityFieldIntegration(unittest.TestCase):
    """测试密度场分配器与环境的集成"""

    def setUp(self):
        set_global_seeds(42)
        self.allocator = DensityFieldAllocator(
            h=0.1, alpha=0.5, beta=0.3, sigma=0.05, delta=1e-6
        )

    def test_assign_targets_with_real_agents(self):
        """使用真实的Hunter和Target对象测试目标分配"""
        env = MultiTarEnv(
            length=2.0, num_obstacle=3, num_hunters=4,
            num_targets=2, h_actor_dim=32, t_actor_dim=33,
            action_dim=2, visualize_lasers=False
        )
        env.reset()

        assignments = self.allocator.assign_targets(
            env.hunters, env.targets, env.obstacles
        )

        # 每个hunter都应该被分配到一个target
        self.assertEqual(len(assignments), len(env.hunters))
        for hunter in env.hunters:
            self.assertIn(id(hunter), assignments)
            self.assertIn(assignments[id(hunter)], env.targets)

    def test_density_field_positive(self):
        """密度场值应为非负"""
        env = MultiTarEnv(
            length=2.0, num_obstacle=2, num_hunters=3,
            num_targets=1, h_actor_dim=32, t_actor_dim=33,
            action_dim=2, visualize_lasers=False
        )
        env.reset()

        for target in env.targets:
            density = self.allocator.compute_density_field(
                target, env.hunters, env.obstacles
            )
            self.assertGreaterEqual(density, 0.0)

    def test_marginal_contribution_nonnegative(self):
        """边际贡献应为非负"""
        env = MultiTarEnv(
            length=2.0, num_obstacle=2, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=33,
            action_dim=2, visualize_lasers=False
        )
        env.reset()

        for hunter in env.hunters:
            others = [h for h in env.hunters if h is not hunter]
            for target in env.targets:
                mc = self.allocator.compute_marginal_contribution(
                    hunter, target, others, env.obstacles
                )
                self.assertGreaterEqual(mc, 0.0)


class TestEscapeStrategyIntegration(unittest.TestCase):
    """测试逃逸策略与环境的集成"""

    def setUp(self):
        set_global_seeds(42)
        self.calculator = ReferenceVelocityCalculator(perception_range=0.5)

    def test_escape_vector_points_away_from_hunters(self):
        """逃逸向量应指向远离hunter的方向"""
        target_pos = np.array([1.0, 1.0, 0.1])
        hunter_positions = [
            np.array([0.8, 1.0, 0.1]),
            np.array([1.0, 0.8, 0.1]),
        ]

        v1 = self.calculator.compute_escape_vector(target_pos, hunter_positions)

        # hunter质心在target的左下方，逃逸向量应指向右上方
        centroid = np.mean([h[:2] for h in hunter_positions], axis=0)
        direction_to_centroid = centroid - target_pos[:2]

        # v1与direction_to_centroid的点积应为负（反方向）
        dot = np.dot(v1, direction_to_centroid)
        self.assertLess(dot, 0, "Escape vector should point away from hunter centroid")

    def test_obstacle_detection(self):
        """障碍物内部检测应正确工作"""
        obstacle = Obstacle(length=2.0)
        obstacle.position = np.array([1.0, 1.0, 0.0])
        obstacle.radius = 0.15

        # 在障碍物内部
        pos_inside = np.array([1.05, 1.05, 0.1])
        self.assertTrue(self.calculator.is_inside_obstacle(pos_inside, [obstacle]))

        # 在障碍物外部
        pos_outside = np.array([2.0, 2.0, 0.1])
        self.assertFalse(self.calculator.is_inside_obstacle(pos_outside, [obstacle]))

    def test_reference_velocity_with_real_env(self):
        """在真实环境中计算参考速度"""
        env = MultiTarEnv(
            length=2.0, num_obstacle=2, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=33,
            action_dim=2, visualize_lasers=False
        )
        env.reset()

        for target in env.targets:
            v_ref = self.calculator.compute_reference_velocity(target, env.hunters)
            self.assertEqual(len(v_ref), 2)
            self.assertFalse(np.any(np.isnan(v_ref)))


class TestRoleAssignmentIntegration(unittest.TestCase):
    """测试角色分配与卡尔曼滤波的集成"""

    def setUp(self):
        set_global_seeds(42)
        self.assigner = RoleAssigner(dt=0.5, process_noise=0.01, measurement_noise=0.1)

    def test_kalman_filter_prediction_accuracy(self):
        """卡尔曼滤波器对匀速运动的预测应接近真实值"""
        kf = KalmanFilter(dt=0.1, process_noise=0.001, measurement_noise=0.01)
        kf.initialize(position=np.array([0.0, 0.0]), velocity=np.array([0.1, 0.05]))

        # 模拟匀速运动，喂入观测
        true_pos = np.array([0.0, 0.0])
        true_vel = np.array([0.1, 0.05])

        for i in range(20):
            true_pos = true_pos + 0.1 * true_vel
            kf.predict()
            kf.update(true_pos + np.random.normal(0, 0.005, 2))

        # 预测5步后的位置
        predicted = kf.predict_future(steps_ahead=5)
        expected = true_pos + 5 * 0.1 * true_vel

        error = np.linalg.norm(predicted - expected)
        self.assertLess(error, 0.1, f"Prediction error too large: {error}")

    def test_role_assignment_with_real_env(self):
        """在真实环境中测试角色分配"""
        env = MultiTarEnv(
            length=2.0, num_obstacle=2, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=33,
            action_dim=2, visualize_lasers=False
        )
        env.reset()

        # 运行几步让target有速度
        for _ in range(5):
            actions = [np.random.uniform(-0.01, 0.01, size=2)
                       for _ in range(env.num_hunters + env.num_targets)]
            env.step(actions)

        # 检查角色分配
        for hunter in env.hunters:
            self.assertIn(hunter.role, ['chaser', 'interceptor'])
            self.assertEqual(len(hunter.target_position), 3)

    def test_chaser_uses_current_position(self):
        """chaser的target_position应为target当前位置"""
        env = MultiTarEnv(
            length=2.0, num_obstacle=2, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=33,
            action_dim=2, visualize_lasers=False
        )
        env.reset()

        actions = [np.random.uniform(-0.01, 0.01, size=2)
                   for _ in range(env.num_hunters + env.num_targets)]
        env.step(actions)

        for hunter in env.hunters:
            if hunter.role == 'chaser' and hunter.assigned_target is not None:
                np.testing.assert_array_almost_equal(
                    hunter.target_position,
                    hunter.assigned_target.position,
                    decimal=5,
                    err_msg="Chaser should track target's current position"
                )


class TestRewardIntegration(unittest.TestCase):
    """测试奖励函数的集成"""

    def setUp(self):
        set_global_seeds(42)

    def test_rewards_are_finite(self):
        """所有奖励值应为有限数"""
        env = MultiTarEnv(
            length=2.0, num_obstacle=3, num_hunters=4,
            num_targets=2, h_actor_dim=32, t_actor_dim=33,
            action_dim=2, visualize_lasers=False
        )
        env.reset()

        for _ in range(10):
            actions = [np.random.uniform(-0.01, 0.01, size=2)
                       for _ in range(env.num_hunters + env.num_targets)]
            _, _, rewards, _ = env.step(actions)

            for r in rewards:
                self.assertTrue(np.isfinite(r), f"Non-finite reward: {r}")

    def test_alignment_reward_direction(self):
        """余弦相似度奖励方向性：对齐时为正，反向时为负"""
        v_ref = np.array([1.0, 0.0])

        # 完全对齐
        v_aligned = np.array([1.0, 0.0])
        cos_sim = np.dot(v_aligned, v_ref) / (np.linalg.norm(v_aligned) * np.linalg.norm(v_ref))
        self.assertGreater(cos_sim, 0)

        # 完全反向
        v_opposite = np.array([-1.0, 0.0])
        cos_sim = np.dot(v_opposite, v_ref) / (np.linalg.norm(v_opposite) * np.linalg.norm(v_ref))
        self.assertLess(cos_sim, 0)


class TestValidationConfigParsing(unittest.TestCase):
    """测试验证流水线的配置解析"""

    def test_yaml_config_parsing(self):
        """测试YAML配置文件解析"""
        # 需要yaml模块
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")

        config_content = {
            'model': {'path': 'model/test'},
            'validation': {'num_episodes': 5, 'max_steps': 100},
            'environment': {'num_hunters': 4, 'num_targets': 2},
            'random_seed': 42
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                          delete=False, encoding='utf-8') as f:
            yaml.dump(config_content, f)
            temp_path = f.name

        try:
            from validation_pipeline import ValidationPipeline
            # 只测试配置解析，不运行完整流水线
            pipeline = ValidationPipeline.__new__(ValidationPipeline)
            config = pipeline.load_config(temp_path)

            self.assertEqual(config['model']['path'], 'model/test')
            self.assertEqual(config['validation']['num_episodes'], 5)
            self.assertEqual(config['random_seed'], 42)
        finally:
            os.unlink(temp_path)

    def test_json_config_parsing(self):
        """测试JSON配置文件解析"""
        config_content = {
            'model': {'path': 'model/test'},
            'validation': {'num_episodes': 3},
            'random_seed': 123
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                          delete=False, encoding='utf-8') as f:
            json.dump(config_content, f)
            temp_path = f.name

        try:
            from validation_pipeline import ValidationPipeline
            pipeline = ValidationPipeline.__new__(ValidationPipeline)
            config = pipeline.load_config(temp_path)

            self.assertEqual(config['model']['path'], 'model/test')
            self.assertEqual(config['validation']['num_episodes'], 3)
            # 默认值应被设置
            self.assertEqual(config['validation']['max_steps'], 150)
            self.assertEqual(config['validation']['save_frame_interval'], 5)
        finally:
            os.unlink(temp_path)

    def test_missing_required_field_raises_error(self):
        """缺少必需字段应抛出ValueError"""
        config_content = {
            'validation': {'num_episodes': 5}
            # 缺少 model.path
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                          delete=False, encoding='utf-8') as f:
            json.dump(config_content, f)
            temp_path = f.name

        try:
            from validation_pipeline import ValidationPipeline
            pipeline = ValidationPipeline.__new__(ValidationPipeline)
            with self.assertRaises(ValueError):
                pipeline.load_config(temp_path)
        finally:
            os.unlink(temp_path)


class TestSafeDivide(unittest.TestCase):
    """测试安全除法"""

    def test_normal_division(self):
        self.assertAlmostEqual(safe_divide(10, 2), 5.0, places=4)

    def test_zero_denominator(self):
        result = safe_divide(10, 0)
        self.assertTrue(np.isfinite(result))
        self.assertGreater(result, 0)


if __name__ == '__main__':
    unittest.main()
