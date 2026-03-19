import numpy as np
import matplotlib.pyplot as plt
import torch as T
import utils
from Lidar import Lidar
from density_field_allocator import DensityFieldAllocator
from reference_velocity_calculator import ReferenceVelocityCalculator
from role_assigner import RoleAssigner

'''
    class MultiTarEnv: is a world created with UAVs (multi-target&hunter) and obstacles in a limited square area.
                       param : 
                            h_actor_dim: a list to store hunters' output dimensions
                            t_actor_dim: a list to store targets' output dimensions
                       funcs :

    class Obstacle: defined obstacles and their contributions
    class Hunter & Target: they'll all be equipped with a sensor to detect obstacles nearby and other UAVs, 
                           and an AI brain(agent network) to make decision, with some elemental contributions.
'''

# add a global random seed to control randomness
def set_global_seeds(seed):
        import random
        np.random.seed(seed)
        random.seed(seed)
        T.manual_seed(seed)

class MultiTarEnv:
    def __init__(self,length,num_obstacle,num_hunters,num_targets,
                 h_actor_dim,t_actor_dim,action_dim,visualize_lasers=False,
                 use_density_field=True, use_role_assignment=True, use_ref_velocity=True):
        self.length = length # length of boundary

        self.num_obstacle = num_obstacle # number of obstacles
        self.num_hunters = num_hunters # number of hunters
        self.num_targets = num_targets # number of targets

        self.h_actor_dim = h_actor_dim # each hunter's observation dimension
        self.t_actor_dim = t_actor_dim # each target's observation dimension
        self.action_dim = action_dim # dimension of each agent's action

        self.time_step = 0.5 # update time step
        self.v_max = 0.05
        self.a_max = 0.01
        self.num_lasers = 16 # beams of lasers
        self.L_sensor = 0.2 # max length of sensor
        self.escape_distance = 0.05 # escape distance threshold for target
        self.distance_threshold = 0.01 # distance threshold for collision
        self.max_escape_angle = 30 # max escape angle for target (degree)

        # 逃逸区域：固定在左下角，target需到达此区域才算逃逸成功
        self.escape_zone_center = np.array([0.15, 0.15])
        self.escape_zone_radius = 0.1
        self.escape_penalty_for_hunters = -5.0  # 逃逸成功时猎手惩罚
        self.escape_reward_for_target = 5.0     # 逃逸成功时目标奖励

        ## Instancing hunters and targets, obstacles.
        # set of obstacles
        self.obstacles = [Obstacle() for _ in range(self.num_obstacle)]
        # set of hunters
        self.hunters = [Hunter(self.length, self.L_sensor,self.num_lasers, self.time_step, self.obstacles) for _ in range(self.num_hunters)]
        # set of targets
        self.targets = [Target(self.length, self.L_sensor,self.num_lasers, self.time_step, self.obstacles) for _ in range(self.num_targets)]

        # Control whether to visualize laser scans & plot relevant
        self.visualize_lasers = visualize_lasers
        self.fig = None
        self.ax = None
        if visualize_lasers:
            self.fig = plt.figure(figsize=(8,8))
            self.ax = self.fig.add_subplot(111,projection='3d')

        # reward relevant
        self.capture_reward = 2.0        # roundup success reward
        self.chase_reward_coeff = 0.8    # chase reward coeff
        self.escape_reward_coeff = 0.4   # escape reward coeff
        self.safe_penalty_coeff = 0.15   # safe penalty coeff (降低，避免惩罚主导)

        # Target reference velocity reward coefficients
        self.alignment_reward_coeff = 0.5  # cosine similarity reward coefficient
        self.obstacle_interior_penalty = 0.3  # penalty for being inside obstacle (降低)
        self.obstacle_proximity_penalty_coeff = 0.6
        
        # Density field allocator for target assignment
        self.density_allocator = DensityFieldAllocator(
            h=0.1,      # kernel bandwidth
            alpha=0.5,  # velocity matching weight
            beta=0.3,   # obstacle attenuation strength
            sigma=0.05, # obstacle influence range
            delta=1e-6  # small constant for utility calculation
        )
        
        # Reference velocity calculator for target escape strategy
        self.ref_velocity_calculator = ReferenceVelocityCalculator(
            perception_range=0.5  # target's perception range for hunters
        )
        
        # Role assigner for hunter role assignment
        self.role_assigner = RoleAssigner(
            dt=self.time_step,
            process_noise=0.01,
            measurement_noise=0.1
        )

        # Ablation switches
        self.use_density_field = use_density_field
        self.use_role_assignment = use_role_assignment
        self.use_ref_velocity = use_ref_velocity
        self.capture_ends_episode = True
        self.current_stage_name = "default"
        self.chase_progress_clip = 2.0 * self.v_max * self.time_step
        self._prev_hunter_positions = {}
        self._prev_target_positions = {}

    def configure_training_phase(self, stage_name=None, reward_config=None, ablation_config=None):
        """
        Update reward weights and ablation switches for curriculum learning.
        """
        if stage_name is not None:
            self.current_stage_name = stage_name

        if reward_config:
            for attr in (
                'capture_reward',
                'chase_reward_coeff',
                'escape_reward_coeff',
                'safe_penalty_coeff',
                'alignment_reward_coeff',
                'obstacle_interior_penalty',
                'obstacle_proximity_penalty_coeff',
                'distance_threshold',
            ):
                if attr in reward_config:
                    setattr(self, attr, reward_config[attr])

        if ablation_config:
            if 'use_density_field' in ablation_config:
                self.use_density_field = ablation_config['use_density_field']
            if 'use_role_assignment' in ablation_config:
                self.use_role_assignment = ablation_config['use_role_assignment']
            if 'use_ref_velocity' in ablation_config:
                self.use_ref_velocity = ablation_config['use_ref_velocity']
    
    def _collect_obs_info(self):
        multi_obs_info = []
        for obstacle in self.obstacles:
            multi_obs_info.append(obstacle._return_obs_info())

    def _compute_hunter_chase_reward(self, hunter, target):
        """
        Distance-progress chase reward gated by heading quality.

        If previous positions are unavailable, fall back to current positions so the
        progress term becomes zero instead of introducing undefined values.
        """
        hunter_pos = hunter.position[:2]
        target_pos = target.position[:2]
        prev_hunter_pos = self._prev_hunter_positions.get(id(hunter), hunter_pos)
        prev_target_pos = self._prev_target_positions.get(id(target), target_pos)

        d_prev = np.linalg.norm(prev_target_pos - prev_hunter_pos)
        d_curr = np.linalg.norm(target_pos - hunter_pos)
        progress = np.clip(d_prev - d_curr, -self.chase_progress_clip, self.chase_progress_clip)

        hunter_dir = hunter.velocity[:2]
        hunter_dir_norm = np.linalg.norm(hunter_dir)
        target_dir = target_pos - hunter_pos
        target_dir_norm = np.linalg.norm(target_dir)
        if hunter_dir_norm > 1e-6 and target_dir_norm > 1e-6:
            cosine_heading = float(np.dot(hunter_dir, target_dir) / (hunter_dir_norm * target_dir_norm))
        else:
            cosine_heading = 0.0

        if progress >= 0.0:
            heading_factor = 0.25 + 0.75 * max(cosine_heading, 0.0)
        else:
            heading_factor = 1.0 + 0.75 * max(-cosine_heading, 0.0)

        chase_reward = float(progress * heading_factor)
        return chase_reward, float(progress), cosine_heading

    def reset(self):
        '''
        Reset the environment to an initial state and returns the initial observations.
        Returns:
            h_obs (list of np.array): Observations for all hunters.
            t_obs (list of np.array): Observations for all targets.
        '''
        # Helper function to check if position is inside any obstacle
        def is_position_valid(position, obstacles, min_clearance=0.2):
            for obstacle in obstacles:
                obs_x, obs_y, obs_z, obs_r, obs_h = obstacle._return_obs_info()
                distance = np.sqrt((position[0] - obs_x)**2 + (position[1] - obs_y)**2)
                if distance < (obs_r + min_clearance):
                    return False
            return True
        
        # Reset hunters with collision avoidance
        for hunter in self.hunters:
            max_attempts = 100
            for attempt in range(max_attempts):
                hunter.position = np.random.uniform(low=0.50, high=0.75, size=(3,))
                hunter.position[-1] = 0.10  # initial height
                if is_position_valid(hunter.position, self.obstacles):
                    break
                if attempt == max_attempts - 1:
                    # Fallback: place in a safe corner
                    hunter.position = np.array([0.1, 0.1, 0.10])
            
            hunter.velocity = np.zeros(3)
            hunter.history_pos = []
            hunter.lasers = hunter.lidar.scan(hunter.position, self.length)

        # Reset targets with collision avoidance
        for target in self.targets:
            max_attempts = 100
            for attempt in range(max_attempts):
                target.position = np.random.uniform(low=1.50, high=1.75, size=(3,))
                target.position[-1] = 0.10  # initial height
                if is_position_valid(target.position, self.obstacles):
                    break
                if attempt == max_attempts - 1:
                    # Fallback: place in a safe corner
                    target.position = np.array([1.9, 1.9, 0.10])
            
            target.velocity = np.zeros(3)
            target.history_pos = []
            target.lasers = target.lidar.scan(target.position, self.length)
            target.reference_velocity = np.zeros(2)  # Initialize reference velocity

        # Reset role assigner (clear Kalman filters)
        self.role_assigner.reset()
        
        # Initialize Kalman filters for all targets
        for target in self.targets:
            target_id = id(target)
            self.role_assigner.update_kalman_filter(target_id, target.position[:2])

        # Assign initial targets to hunters
        self._assign_targets_to_hunters()
        
        # Assign initial roles to hunters
        self._assign_hunter_roles()

        self._prev_hunter_positions = {
            id(hunter): hunter.position[:2].copy() for hunter in self.hunters
        }
        self._prev_target_positions = {
            id(target): target.position[:2].copy() for target in self.targets
        }

        # Get initial observations
        h_obs, t_obs = self._get_observations()

        return h_obs,t_obs

    def step(self,actions):
        """
        Apply actions to agents, update the environment state, compute rewards, and return observations.
        Args:
            actions (list of np.array): Actions for all agents (hunters followed by targets).
        Returns:
            h_obs_next (list of np.array): Next observations for all hunters.
            t_obs_next (list of np.array): Next observations for all targets.
            rewards (list of float): Rewards for all agents.
            dones (list of bool): Done flags for all agents.
        """
        self._prev_hunter_positions = {
            id(hunter): hunter.position[:2].copy() for hunter in self.hunters
        }
        self._prev_target_positions = {
            id(target): target.position[:2].copy() for target in self.targets
        }

        # Apply actions to hunters
        for i, hunter in enumerate(self.hunters):
            hunter.move(actions[i], self.v_max)
        
        # Apply actions to targets
        for i, target in enumerate(self.targets):
            target.move(actions[self.num_hunters + i], self.v_max) # Assume "action" set is combination of hunetrs' & targets'

        # Update lasers after movement
        for hunter in self.hunters:
            hunter.lasers = hunter.lidar.scan(hunter.position, self.length)
        for target in self.targets:
            target.lasers = target.lidar.scan(target.position, self.length)

        # Update reference velocity for each target
        for target in self.targets:
            if self.use_ref_velocity:
                # 完整引导速度：逃逸向量 + 避障向量 + 出口吸引力
                target.reference_velocity = self.ref_velocity_calculator.compute_reference_velocity(
                    target, self.hunters, escape_zone_center=self.escape_zone_center)
            else:
                # 简化引导速度：只有避障斥力 + 随机游走方向（不含逃逸hunter信息）
                v_avoid = self.ref_velocity_calculator.compute_avoidance_vector(
                    target.lasers, target.lidar.angles)
                # 随机游走方向
                angle = np.random.uniform(0, 2 * np.pi)
                v_random = np.array([np.cos(angle), np.sin(angle)])
                # 合成：避障权重大，随机游走权重小
                v_avoid_norm = np.linalg.norm(v_avoid)
                if v_avoid_norm > 1e-6:
                    v_ref = v_avoid / v_avoid_norm + 0.3 * v_random
                else:
                    v_ref = v_random
                target.reference_velocity = v_ref

        # Optionally: Boundary blocking (when train agents, 
        # to allow agents traverse the boundary may get worse performance)
        for agent in self.hunters + self.targets:
            agent.position[:2] = np.clip(agent.position[:2], 0, self.length)

        # Assign targets to hunters using density field allocator
        self._assign_targets_to_hunters()
        
        # Update Kalman filters for all targets
        for target in self.targets:
            target_id = id(target)
            self.role_assigner.update_kalman_filter(target_id, target.position[:2])
        
        # Assign roles to hunters based on their assigned targets
        self._assign_hunter_roles()

        # Compute rewards and check for captures
        rewards, dones, reward_info = self._compute_rewards()

        # Get next observations
        h_obs_next, t_obs_next = self._get_observations()

        return h_obs_next, t_obs_next, rewards, dones, reward_info

    def _assign_targets_to_hunters(self):
        """
        使用密度场分配器或最近距离为hunter分配目标
        """
        if self.use_density_field:
            # 使用密度场分配器计算最优分配
            assignments = self.density_allocator.assign_targets(
                self.hunters,
                self.targets,
                self.obstacles
            )
        else:
            # 消融模式：使用最近距离分配
            assignments = {}
            for hunter in self.hunters:
                nearest = self._get_nearest_target(hunter)
                if nearest is not None:
                    assignments[id(hunter)] = nearest

        # 更新每个hunter的assigned_target属性
        for hunter in self.hunters:
            hunter_id = id(hunter)
            if hunter_id in assignments:
                hunter.assigned_target = assignments[hunter_id]
            else:
                hunter.assigned_target = None
    
    def _assign_hunter_roles(self):
        """
        为hunter分配角色（chaser或interceptor）
        """
        if not self.use_role_assignment:
            # 消融模式：所有hunter均为chaser
            for hunter in self.hunters:
                hunter.role = 'chaser'
                if hunter.assigned_target is not None:
                    hunter.target_position = hunter.assigned_target.position.copy()
                else:
                    hunter.target_position = np.zeros(3)
            return

        # 将hunters按照assigned_target分组
        target_hunter_groups = {}
        for hunter in self.hunters:
            if hunter.assigned_target is not None:
                target = hunter.assigned_target
                if target not in target_hunter_groups:
                    target_hunter_groups[target] = []
                target_hunter_groups[target].append(hunter)
        
        # 为每组hunters分配角色
        for target, hunters in target_hunter_groups.items():
            # 使用role_assigner分配角色
            roles = self.role_assigner.assign_roles(hunters, target)
            
            # 更新每个hunter的role和target_position
            for hunter in hunters:
                hunter_id = id(hunter)
                if hunter_id in roles:
                    hunter.role = roles[hunter_id]
                    # 根据角色设置target_position
                    hunter.target_position = self.role_assigner.get_target_position_for_hunter(
                        hunter, target, hunter.role
                    )
                else:
                    # 默认为chaser
                    hunter.role = 'chaser'
                    hunter.target_position = target.position.copy()
        
        # 处理没有分配到target的hunters
        for hunter in self.hunters:
            if hunter.assigned_target is None:
                hunter.role = 'chaser'
                hunter.target_position = np.zeros(3)

    def _get_nearest_target(self, hunter):
        """
        Find the nearest target to the given hunter.
        Args:
            hunter (Hunter): The hunter to find the nearest target for.
        Returns:
            Target: The nearest target.
        """
        min_dist = float('inf')
        nearest = None
        for target in self.targets:
            dist = np.linalg.norm(hunter.position[:2] - target.position[:2])  # TODO: 2D distance -> 3D
            if dist < min_dist:
                min_dist = dist
                nearest = target
        return nearest
    
    def _get_observations(self):
        """
        Compute observations for all hunters and targets.
        Returns:
            h_obs (list of np.array): Observations for all hunters.
            t_obs (list of np.array): Observations for all targets.
        """
        h_obs = []
        t_obs = []

        # Precompute hunter positions for nearest hunters
        hunter_positions = np.array([hunter.position for hunter in self.hunters])
        hunter_xy = hunter_positions[:, :2]
        if len(hunter_positions) > 1:
            hunter_pairwise = np.linalg.norm(
                hunter_xy[:, None, :] - hunter_xy[None, :, :],
                axis=2,
            )
            np.fill_diagonal(hunter_pairwise, np.inf)
        else:
            hunter_pairwise = None

        target_positions = np.array([target.position for target in self.targets]) if self.targets else np.zeros((0, 3))
        target_xy = target_positions[:, :2] if len(target_positions) else np.zeros((0, 2))
        target_to_hunters = (
            np.linalg.norm(target_xy[:, None, :] - hunter_xy[None, :, :], axis=2)
            if len(target_positions) and len(hunter_positions)
            else np.zeros((len(target_positions), len(hunter_positions)))
        )

        # Compute observations for hunters
        for i, hunter in enumerate(self.hunters):
            if len(hunter_positions) >= 3:
                nearest_indices = np.argpartition(hunter_pairwise[i], 2)[:2]
                nearest_hunters = hunter_positions[nearest_indices]
            else:
                other_hunters = np.delete(hunter_positions, i, axis=0)
                # If less than two other hunters, pad with zeros
                nearest_hunters = np.zeros((2, 3))
                if len(other_hunters) == 1:
                    nearest_hunters[0] = other_hunters[0]
                    nearest_hunters[1] = np.zeros(3)
                else:
                    nearest_hunters = np.zeros((2, 3))

            # Get velocity
            velocity = hunter.velocity

            # Get target position based on hunter's role
            # target_position is already set by _assign_hunter_roles()
            # For chaser: target.position
            # For interceptor: predicted future position
            target_pos = hunter.target_position
            distance_to_target = np.linalg.norm(hunter.position[:2] - target_pos[:2])

            # Get laser data
            laser_data = hunter.lasers  # Assuming it's a 1D array of size num_lasers

            # Concatenate all observation components
            obs = np.concatenate([
                nearest_hunters.flatten()/self.length,                   # 2 * 3 = 6
                hunter.position/self.length,                             # 3
                velocity/self.v_max,                                     # 3
                target_pos/self.length,                                  # 3
                np.array([distance_to_target])/(np.sqrt(2)*self.length), # 1
                laser_data/self.L_sensor                                 # num_lasers
            ]).astype(np.float32)

            h_obs.append(obs)

        # Precompute hunter positions for targets' observations
        for idx, target in enumerate(self.targets):
            # Get target's own position and velocity
            own_pos = target.position
            own_vel = target.velocity

            # Find three nearest hunters
            distances = target_to_hunters[idx]
            nearest_indices = np.argpartition(distances, min(3, len(distances) - 1))[:3] if len(distances) > 0 else []
            nearest_hunters = hunter_positions[nearest_indices] if len(distances) > 0 else np.zeros((0, 3))
            if len(nearest_hunters) < 3:
                # Pad with zeros if less than 3 hunters
                pad_size = 3 - len(nearest_hunters)
                nearest_hunters = np.vstack([nearest_hunters, np.zeros((pad_size, 3))])

            # Get laser data
            laser_data = target.lasers  # Assuming it's a 1D array of size num_lasers
            
            # Get reference velocity
            ref_vel = target.reference_velocity  # 2D reference velocity

            # 逃逸区域方向向量（归一化）
            escape_dir = self.escape_zone_center - target.position[:2]
            escape_dist = np.linalg.norm(escape_dir)
            if escape_dist > 1e-6:
                escape_dir_norm = escape_dir / escape_dist
            else:
                escape_dir_norm = np.zeros(2)

            # Concatenate all observation components
            obs = np.concatenate([
                own_pos/self.length,                    # 3
                own_vel/self.v_max,                     # 3
                nearest_hunters.flatten()/self.length,  # 3 * 3 = 9
                laser_data/self.L_sensor,               # num_lasers
                ref_vel/self.v_max,                     # 2 (reference velocity)
                escape_dir_norm,                        # 2 (escape zone direction)
                np.array([escape_dist / (np.sqrt(2) * self.length)]),  # 1 (escape zone distance)
            ]).astype(np.float32)

            t_obs.append(obs)

        return h_obs, t_obs

    @staticmethod
    def _wrap_angle(angle):
        return angle % (2 * np.pi)

    @staticmethod
    def _shortest_angular_distance(a, b):
        diff = (a - b + np.pi) % (2 * np.pi) - np.pi
        return diff

    def _compute_escape_sector_reward(self, target):
        """
        Reward target motion directions that fall inside the largest clear escape sector.
        The sector is computed with the existing VFH-style histogram, considering hunters,
        boundaries, and obstacles.
        """
        v_actual = target.velocity[:2]
        speed = np.linalg.norm(v_actual)
        if speed < 1e-6:
            return 0.0

        hunter_positions = [tuple(h.position[:2]) for h in self.hunters]
        histogram = utils.compute_histogram(
            tuple(target.position[:2]),
            hunter_positions,
            self.L_sensor * 2,
            max_range=self.L_sensor,
            boundary_length=self.length,
            obstacles=self.obstacles,
        )
        angles = np.arange(0, 2 * np.pi, np.pi / 16)
        largest_escape_interval = utils.find_largest_clear_band(
            histogram, angles, self.L_sensor
        )
        if largest_escape_interval is None:
            return -1.0

        start_deg, end_deg = largest_escape_interval
        start = self._wrap_angle(np.deg2rad(start_deg))
        end = self._wrap_angle(np.deg2rad(end_deg))
        width = (end - start) % (2 * np.pi)
        if width < 1e-6:
            width = 2 * np.pi

        center = self._wrap_angle(start + width / 2.0)
        actual_angle = self._wrap_angle(np.arctan2(v_actual[1], v_actual[0]))
        half_width = max(width / 2.0, 1e-6)
        ratio = abs(self._shortest_angular_distance(actual_angle, center)) / half_width
        normalized_width = float(np.clip(width / (2 * np.pi), 0.0, 1.0))
        # Wide open sectors provide weak directional evidence, so attenuate the
        # reward smoothly instead of hard-clipping it to zero at a threshold.
        sector_weight = float(1.0 / (1.0 + np.exp(8.0 * (normalized_width - 0.55))))
        concentration = 1.0 + 2.5 * sector_weight

        if ratio <= 1.0:
            directional_score = (1.0 - ratio) ** concentration
        else:
            directional_score = -min(ratio - 1.0, 1.0)

        return float(sector_weight * directional_score)

    def _compute_rewards(self):
        """
        Compute rewards for all agents and check for done conditions.
        Returns:
            rewards (list of float): Rewards for all agents (hunters followed by targets).
            dones (list of bool): Done flags for all agents.
            reward_info (dict): Breakdown of reward components for logging.
        """
        rewards = [0.0] * (self.num_hunters + self.num_targets)
        dones = [False] * (self.num_hunters + self.num_targets)

        # Reward component tracking
        chase_rewards = [0.0] * self.num_hunters
        capture_rewards = [0.0] * self.num_hunters
        escape_rewards = [0.0] * self.num_targets
        alignment_rewards = [0.0] * self.num_targets

        # Pre-compute index mappings to avoid repeated .index() calls
        hunter_index_map = {id(h): i for i, h in enumerate(self.hunters)}
        target_index_map = {id(t): i for i, t in enumerate(self.targets)}

        # Map each target to its assigned hunters
        target_hunter_groups = {}
        for target in self.targets:
            target_hunter_groups[target] = [hunter for hunter in self.hunters if hunter.assigned_target == target]

        capture_happened = False
        captured_target_indices = []
        escape_happened = False
        escaped_target_indices = []

        # 检查逃逸区域
        for target in self.targets:
            target_index = target_index_map[id(target)]
            dist_to_escape = np.linalg.norm(target.position[:2] - self.escape_zone_center)
            if dist_to_escape <= self.escape_zone_radius:
                # Target 到达逃逸区域，逃逸成功
                escape_happened = True
                escaped_target_indices.append(target_index)
                rewards[self.num_hunters + target_index] += self.escape_reward_for_target
                dones[self.num_hunters + target_index] = True
                # 所有猎手受惩罚
                for i in range(self.num_hunters):
                    rewards[i] += self.escape_penalty_for_hunters

        # Reward for hunters chasing and capturing targets
        for target, hunters in target_hunter_groups.items():
            # calculate chasing reward and ifrounded reward
            for hunter in hunters:
                chase_reward, _, _ = self._compute_hunter_chase_reward(hunter, target)
                hunter_index = hunter_index_map[id(hunter)]
                rewards[hunter_index] += self.chase_reward_coeff * chase_reward
                chase_rewards[hunter_index] += self.chase_reward_coeff * chase_reward

            multi_hunters_pos = [h.position for h in hunters]
            if utils.isRounded(tuple(target.position[:2]), [tuple(row[:2]) for row in multi_hunters_pos], self.L_sensor, self.max_escape_angle,
                               boundary_length=self.length, obstacles=self.obstacles):
                for hunter in hunters:
                    hunter_index = hunter_index_map[id(hunter)]
                    rewards[hunter_index] += self.capture_reward
                    capture_rewards[hunter_index] += self.capture_reward
                target_index = target_index_map[id(target)]
                dones[self.num_hunters + target_index] = True  # to mark target as done
                capture_happened = True
                captured_target_indices.append(target_index)

        # Reward for targets
        for target in self.targets:
            target_index = target_index_map[id(target)]
            if dones[self.num_hunters + target_index]:
                rewards[self.num_hunters + target_index] += 0  # No additional reward if captured
                continue

            escape_reward = self._compute_escape_sector_reward(target)
            rewards[self.num_hunters + target_index] += self.escape_reward_coeff * escape_reward
            escape_rewards[target_index] += self.escape_reward_coeff * escape_reward

            # Add reference velocity alignment reward
            # Use target.reference_velocity already computed in step() — avoid recomputation
            if self.use_ref_velocity:
                v_ref = target.reference_velocity
            else:
                v_ref = np.zeros(2)
            v_actual = target.velocity[:2]
            
            # Compute cosine similarity: cos(v_actual, v_ref) = (v_actual · v_ref) / (||v_actual|| * ||v_ref||)
            v_actual_norm = np.linalg.norm(v_actual)
            v_ref_norm = np.linalg.norm(v_ref)
            
            epsilon = 1e-6
            if v_actual_norm > epsilon and v_ref_norm > epsilon:
                cosine_similarity = np.dot(v_actual, v_ref) / (v_actual_norm * v_ref_norm)
                alignment_reward = cosine_similarity
            else:
                # If either velocity is zero, no alignment reward
                alignment_reward = 0.0
            
            rewards[self.num_hunters + target_index] += self.alignment_reward_coeff * alignment_reward
            alignment_rewards[target_index] += self.alignment_reward_coeff * alignment_reward

            obstacle_warning_threshold = 0.6 * self.L_sensor
            min_laser_length = float(np.min(target.lasers)) if len(target.lasers) > 0 else self.L_sensor
            if min_laser_length < obstacle_warning_threshold:
                proximity_ratio = (obstacle_warning_threshold - min_laser_length) / max(obstacle_warning_threshold, 1e-6)
                rewards[self.num_hunters + target_index] -= (
                    self.obstacle_proximity_penalty_coeff * proximity_ratio
                )

            # Add obstacle interior penalty
            if self.ref_velocity_calculator.is_inside_obstacle(target.position, self.obstacles):
                rewards[self.num_hunters + target_index] -= self.obstacle_interior_penalty

        # reward for safety (u-u)
        # between hunters
        for i in range(self.num_hunters):
            for j in range(i+1, self.num_hunters):
                distance = np.linalg.norm(self.hunters[i].position[:2] - self.hunters[j].position[:2])
                if distance < self.distance_threshold:
                    penalty = self.safe_penalty_coeff * (self.distance_threshold - distance)
                    rewards[i] -= penalty
                    rewards[j] -= penalty

        # between hunters and targets — 不惩罚hunter靠近target（hunter应主动接近）
        # 只惩罚target被hunter碰撞
        for hunter in self.hunters:
            for target in self.targets:
                distance = np.linalg.norm(hunter.position[:2] - target.position[:2])
                if distance < self.distance_threshold:
                    rewards[self.num_hunters + target_index_map[id(target)]] -= self.safe_penalty_coeff * (self.distance_threshold - distance)
        
        # between targets
        for i in range(self.num_targets):
            for j in range(i+1, self.num_targets):
                distance = np.linalg.norm(self.targets[i].position[:2] - self.targets[j].position[:2])
                if distance < self.distance_threshold:
                    penalty = self.safe_penalty_coeff * (self.distance_threshold - distance)
                    rewards[self.num_hunters + i] -= penalty
                    rewards[self.num_hunters + j] -= penalty

        # reward for safety (uav-obstacles)
        # Only penalize when very close to obstacles (< 50% of sensor range)
        obstacle_danger_threshold = self.L_sensor * 0.5
        for agent in self.hunters + self.targets:
            min_laser_length = min(agent.lasers)
            if min_laser_length < obstacle_danger_threshold:
                # Scaled penalty: 0 at threshold, max at 0
                collision_penalty = -self.safe_penalty_coeff * (obstacle_danger_threshold - min_laser_length) / obstacle_danger_threshold
            else:
                collision_penalty = 0.0
            
            if agent in self.hunters:
                agent_index = hunter_index_map[id(agent)]
            else:
                agent_index = target_index_map[id(agent)] + self.num_hunters
            rewards[agent_index] += collision_penalty

        reward_info = {
            'chase_rewards': chase_rewards,
            'capture_rewards': capture_rewards,
            'escape_rewards': escape_rewards,
            'alignment_rewards': alignment_rewards,
            'capture_happened': capture_happened,
            'captured_targets': captured_target_indices,
            'escape_happened': escape_happened,
            'escaped_targets': escaped_target_indices,
            'stage_name': self.current_stage_name,
        }
        if (capture_happened or escape_happened) and self.capture_ends_episode:
            dones = [True] * (self.num_hunters + self.num_targets)
        return rewards, dones, reward_info
    
    def rewardNorm(self, rewards):
        """
        normalize rewards for hunters and targets
        """
        h_rewards = rewards[:self.num_hunters]
        t_rewards = rewards[self.num_hunters:]
        
        h_mean = np.mean(h_rewards)
        h_std = np.std(h_rewards)
        if h_std == 0:
            h_normalized = h_rewards
        else:
            h_normalized = (h_rewards - h_mean) / h_std
        
        t_mean = np.mean(t_rewards)
        t_std = np.std(t_rewards)
        if t_std == 0:
            t_normalized = t_rewards
        else:
            t_normalized = (t_rewards - t_mean) / t_std
        
        normalized_rewards = list(h_normalized) + list(t_normalized)
        return normalized_rewards

    # TODO: use simulator like airsim to render
    def render(self):
        """
        Visualize the environment.
        """
        # Lazy initialization of figure
        if self.fig is None:
            self.fig = plt.figure(figsize=(8,8))
            self.ax = self.fig.add_subplot(111,projection='3d')
        
        self.ax.clear()
        self.ax.set_xlim(0, self.length)
        self.ax.set_ylim(0, self.length)
        self.ax.set_zlim(0, self.length/4)
        self.ax.set_title("环境可视化")

        # Draw boundaries
        self.ax.plot([0, self.length, self.length, 0, 0], [0, 0, self.length, self.length, 0], [0, 0, 0, 0, 0], color='black', linewidth=2)
        # Draw obstacles (cylinders)
        for obstacle in self.obstacles:
            cx, cy, cz, r, h = obstacle._return_obs_info()
            self._create_cylinders(self.ax, cx, cy, cz, r, h)

        # Draw hunters
        for hunter in self.hunters:
            x, y, z = hunter.position
            self.ax.scatter(x, y, z, color='red', label='追击者' if hunter == self.hunters[0] else "")
            if self.visualize_lasers:
                # Draw lasers
                hunter.lidar.visualize_lasers(hunter.position,self.ax)

        # Draw targets
        for target in self.targets:
            x, y, z = target.position
            self.ax.scatter(x, y, z, color='green', label='逃逸者' if target == self.targets[0] else "")

        self.ax.legend(loc='upper right')
        plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


    def _create_cylinders(self, ax, x, y, z, r, h):
        """
        Create a 3D cylinder to represent obstacles in the environment.
        """
        # Create obstacle as a cylinder in 3D
        # reduce the number of sampling points in the circumferential and height directions 
        # to improve rendering performance
        theta = np.linspace(0, 2 * np.pi, 16) 
        z_vals = np.linspace(z, z + h, 8)   
        theta, z_vals = np.meshgrid(theta, z_vals)
        x_vals = x + r * np.cos(theta)
        y_vals = y + r * np.sin(theta)

        ax.plot_surface(x_vals, y_vals, z_vals, color='black', alpha=0.5)

class Obstacle:
    def __init__(self, length=2, speed=0):
        self.position = np.random.uniform(low=0.45, high=length-0.55, size=(3,))
        self.position[-1] = 0 # firstly let z = 0
        self.speed = speed
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = np.array([self.speed * np.cos(angle), self.speed * np.sin(angle)])
        self.radius = np.random.uniform(0.1, 0.15)
        self.height = np.random.uniform(0.1, 0.15)

    # TODO: make movable obstacles update position
    def move_obs(self):
        pass

    # to depackage obstacles' information
    def _return_obs_info(self):
        x,y,z = self.position
        r = self.radius
        h = self.height
        return (x,y,z,r,h)

class AgentBase:
    def __init__(self, boundary_length, max_distance=0.2, num_rays=16, time_step=0.5, obstacles=None):
        self.boundary_length = boundary_length
        self.position = np.random.uniform(low=1.25, high=1.5, size=(3,))  # TODO: initial spawn scope
        self.position[-1] = 0.10  # initial height
        self.velocity = np.zeros(3)  # initial velocity
        self.time_step = time_step  # update time step

        self.lidar = Lidar(max_distance, num_rays, obstacles)
        self.lasers = self.lidar.scan(self.position, self.boundary_length)
        self.lidar.distances = self.lasers
        self.history_pos = []  # to store trajectory

    # update state
    def move(self, action, v_max=0.1):
        ax, ay = action
        if self.lidar.isInObs:
            self.velocity = np.zeros(3)
        else:
            self.velocity[0] += self.time_step * ax
            self.velocity[1] += self.time_step * ay
            velocity_norm = (self.velocity[0]**2 + self.velocity[1]**2)**0.5
            if velocity_norm > v_max:
                scale_factor = v_max / velocity_norm
                self.velocity[0] *= scale_factor
                self.velocity[1] *= scale_factor

        self.position += self.time_step * self.velocity 
        self.lasers = self.lidar.scan(self.position, self.boundary_length)  
        self.history_pos.append(self.position.copy())

class Hunter(AgentBase):
    def __init__(self, boundary_length, max_distance=0.2, num_rays=16, time_step=0.5, obstacles=None):
        super().__init__(boundary_length, max_distance, num_rays, time_step, obstacles)  # inherit base class
        self.assigned_target = None  # Target assigned by density field allocator
        self.role = 'chaser'  # Role: 'chaser' or 'interceptor'
        self.target_position = np.zeros(3)  # Current target position to pursue (may be predicted position)
        '''
            # TODO: define hunter's role, chaser/predator, chaser will chase target's current position, 
                    while predator will use KF (or other way) to predict target's future position and head to it.
                    so difference between these two roles lies in input.
        '''

class Target(AgentBase):
    def __init__(self, boundary_length, max_distance=0.2, num_rays=16, time_step=0.5, obstacles=None):
        super().__init__(boundary_length, max_distance, num_rays, time_step, obstacles)  # inherit base class
        self.reference_velocity = np.zeros(2)  # Reference velocity for escape strategy
