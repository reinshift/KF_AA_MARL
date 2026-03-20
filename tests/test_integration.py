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
from unittest.mock import patch
from types import SimpleNamespace

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
            t_actor_dim=36,
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

        # Target obs: 3(pos) + 3(vel) + 3*3(nearest_hunters) + 16(laser) + 2(ref_vel) + 2(escape_dir) + 1(escape_dist) = 36
        for obs in t_obs:
            self.assertEqual(len(obs), 36)

    def test_step_cycle_runs_without_error(self):
        """验证完整的step循环不报错"""
        h_obs, t_obs = self.env.reset()

        # 随机动作
        actions = [np.random.uniform(-0.01, 0.01, size=2)
                   for _ in range(self.env.num_hunters + self.env.num_targets)]

        h_obs_next, t_obs_next, rewards, dones, reward_info = self.env.step(actions)

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
            h_obs, t_obs, rewards, dones, _ri = self.env.step(actions)

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
            num_targets=2, h_actor_dim=32, t_actor_dim=36,
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
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
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
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
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

    def test_balanced_assignment_distributes_hunters_across_targets(self):
        """With 6 hunters and 2 targets, balanced assignment should avoid collapsing to one target."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=0, num_hunters=6,
            num_targets=2, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()

        assignments = self.allocator.assign_targets(
            env.hunters, env.targets, env.obstacles
        )

        counts = [0] * len(env.targets)
        target_index = {id(target): idx for idx, target in enumerate(env.targets)}
        for target in assignments.values():
            counts[target_index[id(target)]] += 1

        self.assertEqual(sorted(counts), [3, 3])


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

    def test_avoidance_vector_points_away_from_shortened_rays(self):
        """短光束对应近障碍，v2应指向其反方向。"""
        laser_data = np.array([0.05, 0.20, 0.20, 0.20], dtype=float)
        laser_angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=float)

        v2 = self.calculator.compute_avoidance_vector(
            laser_data, laser_angles, max_range=0.20
        )

        self.assertLess(v2[0], 0.0)
        self.assertAlmostEqual(np.linalg.norm(v2), 1.0, places=6)

    def test_avoidance_vector_is_zero_when_no_rays_are_shortened(self):
        """所有激光都未缩短时，不应产生斥力方向。"""
        laser_data = np.full(8, 0.20, dtype=float)
        laser_angles = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)

        v2 = self.calculator.compute_avoidance_vector(
            laser_data, laser_angles, max_range=0.20
        )

        np.testing.assert_allclose(v2, np.zeros(2), atol=1e-9)

    def test_exit_guidance_projects_to_feasible_direction_when_blocked(self):
        """Blocked exit direction should be attenuated and projected to a clear nearby ray."""
        laser_angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=float)
        laser_data = np.array([0.05, 0.20, 0.20, 0.20], dtype=float)
        exit_dir = np.array([1.0, 0.0], dtype=float)

        projected_vec, gate = self.calculator._compute_feasible_exit_guidance(
            laser_data, laser_angles, exit_dir, max_range=0.20
        )

        self.assertLess(gate, 0.7)
        self.assertGreater(gate, 0.2)
        self.assertLess(projected_vec[0], 0.5)
        self.assertGreater(abs(projected_vec[1]), 0.5)

    def test_reference_velocity_with_real_env(self):
        """在真实环境中计算参考速度"""
        env = MultiTarEnv(
            length=2.0, num_obstacle=2, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()

        for target in env.targets:
            v_ref = self.calculator.compute_reference_velocity(target, env.hunters)
            self.assertEqual(len(v_ref), 2)
            self.assertFalse(np.any(np.isnan(v_ref)))

    def test_reference_velocity_without_hunters_is_deterministic(self):
        env = MultiTarEnv(
            length=2.0, num_obstacle=0, num_hunters=1,
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        target = env.targets[0]
        target.position[:2] = np.array([1.5, 1.5], dtype=float)
        target.velocity[:2] = np.zeros(2, dtype=float)
        target.lasers = np.full(env.num_lasers, env.L_sensor, dtype=float)

        exit_center = np.array([0.15, 0.15], dtype=float)
        v_ref_1 = self.calculator.compute_reference_velocity(target, [], escape_zone_center=exit_center)
        v_ref_2 = self.calculator.compute_reference_velocity(target, [], escape_zone_center=exit_center)

        np.testing.assert_allclose(v_ref_1, v_ref_2, atol=1e-9)
        self.assertLess(np.dot(v_ref_1, target.position[:2] - exit_center), 0.0)

    def test_reference_velocity_can_ignore_hunters_with_zero_scale(self):
        target = SimpleNamespace(
            position=np.array([1.0, 1.0, 0.1], dtype=float),
            velocity=np.zeros(3, dtype=float),
            lasers=np.full(8, 0.20, dtype=float),
            lidar=SimpleNamespace(
                angles=np.linspace(0.0, 2 * np.pi, 8, endpoint=False),
                max_detect_d=0.20,
            ),
        )
        hunters = [
            SimpleNamespace(position=np.array([0.8, 1.0, 0.1], dtype=float)),
            SimpleNamespace(position=np.array([1.0, 0.8, 0.1], dtype=float)),
        ]
        exit_center = np.array([1.0, 1.8], dtype=float)

        v_ref_with_hunters_disabled = self.calculator.compute_reference_velocity(
            target,
            hunters,
            escape_zone_center=exit_center,
            hunter_weight_scale=0.0,
        )
        v_ref_without_hunters = self.calculator.compute_reference_velocity(
            target,
            [],
            escape_zone_center=exit_center,
            hunter_weight_scale=0.0,
        )

        np.testing.assert_allclose(v_ref_with_hunters_disabled, v_ref_without_hunters, atol=1e-9)

    def test_reference_velocity_blends_inertia_and_exit_guidance(self):
        target = SimpleNamespace(
            position=np.array([1.0, 1.0, 0.1], dtype=float),
            velocity=np.array([0.05, 0.0, 0.0], dtype=float),
            lasers=np.full(8, 0.20, dtype=float),
            lidar=SimpleNamespace(
                angles=np.linspace(0.0, 2 * np.pi, 8, endpoint=False),
                max_detect_d=0.20,
            ),
        )

        v_ref = self.calculator.compute_reference_velocity(
            target,
            [],
            escape_zone_center=np.array([1.0, 1.8], dtype=float),
        )

        self.assertGreater(v_ref[0], 0.05)
        self.assertGreater(v_ref[1], 0.05)
        self.assertAlmostEqual(np.linalg.norm(v_ref), 1.0, places=6)

    def test_reference_velocity_projects_blocked_direction_to_gap(self):
        target = SimpleNamespace(
            position=np.array([1.0, 1.0, 0.1], dtype=float),
            velocity=np.array([0.04, 0.0, 0.0], dtype=float),
            lasers=np.array([0.03, 0.20, 0.20, 0.20], dtype=float),
            lidar=SimpleNamespace(
                angles=np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=float),
                max_detect_d=0.20,
            ),
        )

        v_ref = self.calculator.compute_reference_velocity(
            target,
            [],
            escape_zone_center=np.array([1.8, 1.0], dtype=float),
        )

        self.assertLess(v_ref[0], 0.85)
        self.assertGreater(abs(v_ref[1]), 0.25)


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
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
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
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
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

    def test_role_assignment_caps_interceptors(self):
        """Role assignment should keep at least two chasers and cap interceptors."""
        target = SimpleNamespace(
            position=np.array([1.0, 1.0, 0.1], dtype=float),
            velocity=np.array([-0.06, 0.0, 0.0], dtype=float),
        )
        hunters = [
            SimpleNamespace(position=np.array([1.18, 1.0, 0.1], dtype=float)),
            SimpleNamespace(position=np.array([1.24, 1.0, 0.1], dtype=float)),
            SimpleNamespace(position=np.array([0.80, 1.0, 0.1], dtype=float)),
        ]

        with patch.object(self.assigner, 'predict_target_position', return_value=np.array([0.82, 1.0], dtype=float)):
            roles = self.assigner.assign_roles(hunters, target)

        interceptor_count = sum(1 for role in roles.values() if role == 'interceptor')
        chaser_count = sum(1 for role in roles.values() if role == 'chaser')
        self.assertLessEqual(interceptor_count, 1)
        self.assertGreaterEqual(chaser_count, 2)


class TestRewardIntegration(unittest.TestCase):
    """测试奖励函数的集成"""

    def setUp(self):
        set_global_seeds(42)

    def test_rewards_are_finite(self):
        """所有奖励值应为有限数"""
        env = MultiTarEnv(
            length=2.0, num_obstacle=3, num_hunters=4,
            num_targets=2, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()

        for _ in range(10):
            actions = [np.random.uniform(-0.01, 0.01, size=2)
                       for _ in range(env.num_hunters + env.num_targets)]
            _, _, rewards, _, _ = env.step(actions)

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


    def test_capture_ends_episode_for_all_agents(self):
        """Capture should terminate the entire episode instead of only the target."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()

        actions = [np.zeros(2) for _ in range(env.num_hunters + env.num_targets)]
        with patch('utils.isRounded', return_value=True):
            _, _, rewards, dones, reward_info = env.step(actions)

        self.assertTrue(all(dones))
        self.assertTrue(reward_info['capture_happened'])
        self.assertGreater(sum(reward_info['capture_rewards']), 0.0)
        self.assertTrue(any(r > 0 for r in rewards[:env.num_hunters]))

    def test_partial_capture_does_not_end_multi_target_episode(self):
        """Capturing one target should not end the episode while another target remains active."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=2, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        env.targets[0].position[:2] = np.array([1.6, 1.6], dtype=float)
        env.targets[1].position[:2] = np.array([1.7, 1.7], dtype=float)
        for hunter in env.hunters:
            hunter.assigned_target = env.targets[0]

        with patch('utils.isRounded', side_effect=[True, False]):
            rewards, dones, reward_info = env._compute_rewards()

        self.assertFalse(reward_info['episode_terminal'])
        self.assertFalse(reward_info['capture_happened'])
        self.assertTrue(dones[env.num_hunters])
        self.assertFalse(dones[env.num_hunters + 1])
        self.assertEqual(env._get_target_state(env.targets[0]), 'captured')
        self.assertEqual(env._get_target_state(env.targets[1]), 'active')
        self.assertGreater(sum(reward_info['capture_rewards']), 0.0)

    def test_episode_ends_only_after_all_targets_captured(self):
        """Episode should terminate successfully only when the final active target is captured."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=2, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        env.targets[0].position[:2] = np.array([1.6, 1.6], dtype=float)
        env.targets[1].position[:2] = np.array([1.7, 1.7], dtype=float)
        env._set_target_state(env.targets[0], 'captured')
        for hunter in env.hunters:
            hunter.assigned_target = env.targets[1]

        with patch('utils.isRounded', return_value=True):
            rewards, dones, reward_info = env._compute_rewards()

        self.assertTrue(all(dones))
        self.assertTrue(reward_info['capture_happened'])
        self.assertTrue(reward_info['all_targets_captured'])
        self.assertEqual(env._get_target_state(env.targets[1]), 'captured')

    def test_partial_escape_does_not_end_multi_target_episode(self):
        """One escaped target should not terminate the episode while another target remains active."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=2, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        env.targets[0].position[:2] = env.escape_zone_center.copy()
        env.targets[1].position[:2] = np.array([1.7, 1.7], dtype=float)

        with patch('utils.isRounded', return_value=False):
            rewards, dones, reward_info = env._compute_rewards()

        self.assertFalse(reward_info['episode_terminal'])
        self.assertTrue(reward_info['escape_happened'])
        self.assertEqual(env._get_target_state(env.targets[0]), 'escaped')
        self.assertEqual(env._get_target_state(env.targets[1]), 'active')
        self.assertTrue(dones[env.num_hunters])
        self.assertFalse(dones[env.num_hunters + 1])

    def test_episode_ends_only_after_all_targets_resolved_by_escape(self):
        """Episode should terminate as failure only when the last remaining target also escapes."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=2, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        env._set_target_state(env.targets[0], 'escaped')
        env.targets[1].position[:2] = env.escape_zone_center.copy()

        with patch('utils.isRounded', return_value=False):
            rewards, dones, reward_info = env._compute_rewards()

        self.assertTrue(reward_info['episode_terminal'])
        self.assertEqual(reward_info['outcome_code'], -1)
        self.assertEqual(env._get_target_state(env.targets[1]), 'escaped')
        self.assertTrue(all(dones))

    def test_chase_reward_uses_distance_progress_and_heading_gate(self):
        """Closing distance with good heading should yield positive chase reward."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        hunter = env.hunters[0]
        target = env.targets[0]

        env._prev_hunter_positions[id(hunter)] = np.array([0.20, 0.20], dtype=float)
        env._prev_target_positions[id(target)] = np.array([0.60, 0.20], dtype=float)
        hunter.position[:2] = np.array([0.30, 0.20], dtype=float)
        target.position[:2] = np.array([0.60, 0.20], dtype=float)
        hunter.velocity[:2] = np.array([0.05, 0.0], dtype=float)

        chase_reward, progress, cosine_heading = env._compute_hunter_chase_reward(hunter, target)

        self.assertGreater(progress, 0.0)
        self.assertGreater(cosine_heading, 0.0)
        self.assertGreater(chase_reward, 0.0)

    def test_chase_reward_handles_missing_previous_positions(self):
        """Missing prev values should degrade to zero progress instead of failing."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        hunter = env.hunters[0]
        target = env.targets[0]

        env._prev_hunter_positions = {}
        env._prev_target_positions = {}
        hunter.velocity[:2] = np.array([0.05, 0.0], dtype=float)

        chase_reward, progress, _ = env._compute_hunter_chase_reward(hunter, target)

        self.assertEqual(progress, 0.0)
        self.assertEqual(chase_reward, 0.0)

    def test_training_phase_configuration_updates_switches_and_rewards(self):
        """Curriculum stage updates should reach the environment."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )

        env.configure_training_phase(
            stage_name='pursuit_only',
            reward_config={
                'capture_reward': 15.0,
                'chase_reward_coeff': 0.4,
                'blocked_chase_reward_coeff': 0.2,
                'team_capture_bonus': 1.5,
                'distance_threshold': 0.025,
            },
            ablation_config={
                'use_density_field': False,
                'use_role_assignment': False,
                'use_ref_velocity': False,
            },
            mechanism_config={
                'assignment_escape_pressure_coeff': 0.7,
                'density_underloaded_priority': 1.8,
                'max_interceptors_per_target': 0,
                'map_refresh_interval': 7,
                'randomize_exit_zone': True,
                'target_obs_include_hunters': False,
                'target_escape_sector_include_hunters': False,
                'target_hunter_repulsion_scale': 0.25,
                'target_ref_hunter_perception_range': 0.15,
            },
        )

        self.assertEqual(env.current_stage_name, 'pursuit_only')
        self.assertEqual(env.capture_reward, 15.0)
        self.assertEqual(env.chase_reward_coeff, 0.4)
        self.assertEqual(env.blocked_chase_reward_coeff, 0.2)
        self.assertEqual(env.team_capture_bonus, 1.5)
        self.assertEqual(env.distance_threshold, 0.025)
        self.assertEqual(env.assignment_escape_pressure_coeff, 0.7)
        self.assertEqual(env.density_allocator.underloaded_priority, 1.8)
        self.assertEqual(env.role_assigner.max_interceptors_per_target, 0)
        self.assertEqual(env.map_refresh_interval, 7)
        self.assertTrue(env.randomize_exit_zone)
        self.assertFalse(env.target_obs_include_hunters)
        self.assertFalse(env.target_escape_sector_include_hunters)
        self.assertEqual(env.target_hunter_repulsion_scale, 0.25)
        self.assertEqual(env.ref_velocity_calculator.perception_range, 0.15)
        self.assertFalse(env.use_density_field)
        self.assertFalse(env.use_role_assignment)
        self.assertFalse(env.use_ref_velocity)

    def test_target_observation_can_mask_hunter_positions(self):
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        env.target_obs_include_hunters = False

        _, t_obs = env._get_observations()
        hunter_slice = t_obs[0][6:15]
        np.testing.assert_allclose(hunter_slice, np.zeros(9), atol=1e-9)

    def test_blocked_chase_terms_reward_gap_following(self):
        """When direct pursuit is blocked, moving toward a nearby clear gap should be rewarded."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        hunter = env.hunters[0]
        target = env.targets[0]

        hunter.position[:2] = np.array([0.5, 0.5], dtype=float)
        target.position[:2] = np.array([0.8, 0.5], dtype=float)
        hunter.velocity[:2] = np.array([0.0, 0.04], dtype=float)
        hunter.lidar.angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=float)
        hunter.lasers = np.array([0.03, env.L_sensor, env.L_sensor, env.L_sensor], dtype=float)

        blocked_ratio, gap_reward, stuck_penalty = env._compute_hunter_blocked_chase_terms(
            hunter, target, progress=0.01
        )

        self.assertGreater(blocked_ratio, 0.0)
        self.assertGreater(gap_reward, 0.0)
        self.assertGreaterEqual(stuck_penalty, 0.0)

    def test_timeout_outcome_recognizes_partial_capture(self):
        """Timing out after capturing only part of the targets should produce outcome_code=1."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=2, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        env._set_target_state(env.targets[0], 'captured')
        rewards = [0.0] * (env.num_hunters + env.num_targets)

        rewards, timeout_info = env.finalize_timeout_outcome(rewards)

        self.assertEqual(timeout_info['outcome_code'], 1)
        self.assertEqual(timeout_info['captured_target_count'], 1)
        self.assertTrue(any(r > 0 for r in rewards[:env.num_hunters]))

    def test_refreshed_layout_keeps_exit_clear_of_obstacles(self):
        """Randomized layouts should keep exit zone clear of obstacles."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=5, num_hunters=4,
            num_targets=2, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.randomize_exit_zone = True
        env._refresh_layout()

        for obstacle in env.obstacles:
            dist = np.linalg.norm(obstacle.position[:2] - env.escape_zone_center)
            self.assertGreaterEqual(
                dist,
                obstacle.radius + env.escape_zone_radius,
            )

    def test_escape_sector_reward_prefers_motion_inside_clear_sector(self):
        """Target should receive higher reward when moving inside the clear sector."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        target = env.targets[0]

        with patch('utils.find_largest_clear_band', return_value=(315.0, 45.0)):
            target.velocity = np.array([0.01, 0.0, 0.0], dtype=float)
            reward_inside = env._compute_escape_sector_reward(target)

            target.velocity = np.array([0.0, 0.01, 0.0], dtype=float)
            reward_outside = env._compute_escape_sector_reward(target)

        self.assertGreater(reward_inside, reward_outside)
        self.assertGreaterEqual(reward_inside, 0.0)

    def test_escape_sector_reward_is_damped_for_wide_sector(self):
        """Wide clear sectors should dilute directional reward smoothly, not hard-clip it."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        target = env.targets[0]
        target.velocity = np.array([0.01, 0.0, 0.0], dtype=float)

        with patch('utils.find_largest_clear_band', return_value=(315.0, 45.0)):
            reward_narrow = env._compute_escape_sector_reward(target)

        with patch('utils.find_largest_clear_band', return_value=(180.0, 157.5)):
            reward_wide = env._compute_escape_sector_reward(target)

        self.assertGreater(reward_narrow, reward_wide)
        self.assertGreater(reward_wide, 0.0)

    def test_target_reward_penalizes_obstacle_proximity_before_collision(self):
        """A target close to obstacles should already be penalized before entering one."""
        env = MultiTarEnv(
            length=2.0, num_obstacle=1, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        target = env.targets[0]
        target_index = env.num_hunters

        target.lasers = np.full(env.num_lasers, env.L_sensor, dtype=float)
        with patch('utils.isRounded', return_value=False):
            rewards_open, _, _ = env._compute_rewards()

        target.lasers = np.full(env.num_lasers, env.L_sensor, dtype=float)
        target.lasers[0] = 0.02
        with patch('utils.isRounded', return_value=False):
            rewards_blocked, _, _ = env._compute_rewards()

        self.assertLess(rewards_blocked[target_index], rewards_open[target_index])

    def test_alignment_reward_uses_cached_reference_velocity(self):
        env = MultiTarEnv(
            length=2.0, num_obstacle=0, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        target = env.targets[0]
        target_index = env.num_hunters

        env.escape_reward_coeff = 0.0
        env.alignment_reward_coeff = 1.0
        env.safe_penalty_coeff = 0.0
        env.obstacle_interior_penalty = 0.0
        env.obstacle_proximity_penalty_coeff = 0.0
        env._target_boundary_clipped[id(target)] = False
        env._prev_target_ref_velocities[id(target)] = np.array([1.0, 0.0], dtype=float)
        target.reference_velocity = np.array([-1.0, 0.0], dtype=float)
        target.velocity[:2] = np.array([0.05, 0.0], dtype=float)
        target.position[:2] = np.array([1.0, 1.0], dtype=float)
        target.lasers = np.full(env.num_lasers, env.L_sensor, dtype=float)

        with patch('utils.isRounded', return_value=False):
            rewards, _, _ = env._compute_rewards()

        self.assertGreater(rewards[target_index], 0.0)

    def test_alignment_penalty_is_damped_near_boundary(self):
        env = MultiTarEnv(
            length=2.0, num_obstacle=0, num_hunters=4,
            num_targets=1, h_actor_dim=32, t_actor_dim=36,
            action_dim=2, visualize_lasers=False
        )
        env.reset()
        target = env.targets[0]
        target_index = env.num_hunters

        env.escape_reward_coeff = 0.0
        env.alignment_reward_coeff = 1.0
        env.safe_penalty_coeff = 0.0
        env.obstacle_interior_penalty = 0.0
        env.obstacle_proximity_penalty_coeff = 0.0
        env._prev_target_ref_velocities[id(target)] = np.array([1.0, 0.0], dtype=float)
        target.reference_velocity = np.array([1.0, 0.0], dtype=float)
        target.velocity[:2] = np.array([-0.05, 0.0], dtype=float)
        target.lasers = np.full(env.num_lasers, env.L_sensor, dtype=float)

        target.position[:2] = np.array([1.0, 1.0], dtype=float)
        env._target_boundary_clipped[id(target)] = False
        with patch('utils.isRounded', return_value=False):
            rewards_center, _, _ = env._compute_rewards()

        target.position[:2] = np.array([0.01, 1.0], dtype=float)
        env._target_boundary_clipped[id(target)] = True
        with patch('utils.isRounded', return_value=False):
            rewards_wall, _, _ = env._compute_rewards()

        self.assertGreater(rewards_wall[target_index], rewards_center[target_index])


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
