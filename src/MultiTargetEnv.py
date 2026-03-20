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
        self.hunter_spawn_low = 0.50
        self.hunter_spawn_high = 0.75
        self.target_spawn_low = 1.50
        self.target_spawn_high = 1.75

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
        self.team_capture_bonus = 0.0
        self.hunter_time_penalty_coeff = 0.0
        self.blocked_chase_reward_coeff = 0.0
        self.stuck_penalty_coeff = 0.0
        self.containment_progress_reward_coeff = 0.0
        self.containment_quality_reward_coeff = 0.0
        self.chaser_slot_reward_coeff = 0.0
        self.chaser_side_balance_reward_coeff = 0.0
        self.interceptor_quality_reward_coeff = 0.0
        self.full_capture_outcome_reward = 10.0
        self.partial_capture_outcome_reward = 3.0
        self.timeout_outcome_reward = -1.0
        self.escape_outcome_reward = -6.0
        self.target_full_capture_outcome_reward = -4.0
        self.target_partial_capture_outcome_reward = -1.5
        self.target_timeout_outcome_reward = 0.5
        self.target_escape_outcome_reward = 6.0

        # Target reference velocity reward coefficients
        self.alignment_reward_coeff = 0.5  # cosine similarity reward coefficient
        self.obstacle_interior_penalty = 0.3  # penalty for being inside obstacle (降低)
        self.obstacle_proximity_penalty_coeff = 0.6
        self.target_obstacle_warning_ratio = 0.35
        self.obstacle_danger_ratio = 0.35
        self.target_obs_include_hunters = True
        self.target_escape_sector_include_hunters = True
        self.target_hunter_repulsion_scale = 1.0
        self.target_hunter_contact_penalty_coeff = 1.0
        self.assignment_escape_pressure_coeff = 0.0
        self.randomize_layout = True
        self.randomize_exit_zone = False
        self.map_refresh_interval = 20
        self.layout_clearance_margin = 0.08
        self.spawn_clearance_margin = 0.12
        self.exit_clearance_margin = 0.10
        self.intercept_projection_clearance = 0.04
        self.containment_slot_forward_offset = 0.06
        self.containment_slot_lateral_offset = 0.09

        # Density field allocator for target assignment
        self.density_allocator = DensityFieldAllocator(
            h=0.1,      # kernel bandwidth
            alpha=0.5,  # velocity matching weight
            beta=0.3,   # obstacle attenuation strength
            sigma=0.05, # obstacle influence range
            delta=1e-6,  # small constant for utility calculation
            underloaded_priority=1.2,
            over_assignment_penalty=0.8,
        )
        
        # Reference velocity calculator for target escape strategy
        self.ref_velocity_calculator = ReferenceVelocityCalculator(
            perception_range=0.5  # target's perception range for hunters
        )
        
        # Role assigner for hunter role assignment
        self.role_assigner = RoleAssigner(
            dt=self.time_step,
            process_noise=0.01,
            measurement_noise=0.1,
            min_group_size_for_interceptor=3,
            max_interceptors_per_target=1,
            min_target_speed=0.025,
            min_interceptor_distance=0.12,
            max_interceptor_distance=0.30,
            prediction_steps=4,
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
        self._prev_hunter_objectives = {}
        self._prev_target_ref_velocities = {}
        self._prev_target_escape_bandwidths = {}
        self._target_boundary_clipped = {}
        self.target_states = {}
        self._reset_count = 0

    def configure_training_phase(self, stage_name=None, reward_config=None, ablation_config=None,
                                 mechanism_config=None):
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
                'target_obstacle_warning_ratio',
                'obstacle_danger_ratio',
                'target_hunter_contact_penalty_coeff',
                'team_capture_bonus',
                'hunter_time_penalty_coeff',
                'blocked_chase_reward_coeff',
                'stuck_penalty_coeff',
                'containment_progress_reward_coeff',
                'containment_quality_reward_coeff',
                'chaser_slot_reward_coeff',
                'chaser_side_balance_reward_coeff',
                'interceptor_quality_reward_coeff',
                'distance_threshold',
                'full_capture_outcome_reward',
                'partial_capture_outcome_reward',
                'timeout_outcome_reward',
                'escape_outcome_reward',
                'target_full_capture_outcome_reward',
                'target_partial_capture_outcome_reward',
                'target_timeout_outcome_reward',
                'target_escape_outcome_reward',
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

        if mechanism_config:
            if 'assignment_escape_pressure_coeff' in mechanism_config:
                self.assignment_escape_pressure_coeff = mechanism_config['assignment_escape_pressure_coeff']
            if 'density_underloaded_priority' in mechanism_config:
                self.density_allocator.underloaded_priority = mechanism_config['density_underloaded_priority']
            if 'density_over_assignment_penalty' in mechanism_config:
                self.density_allocator.over_assignment_penalty = mechanism_config['density_over_assignment_penalty']
            if 'min_group_size_for_interceptor' in mechanism_config:
                self.role_assigner.min_group_size_for_interceptor = mechanism_config['min_group_size_for_interceptor']
            if 'max_interceptors_per_target' in mechanism_config:
                self.role_assigner.max_interceptors_per_target = mechanism_config['max_interceptors_per_target']
            if 'min_target_speed_for_interceptor' in mechanism_config:
                self.role_assigner.min_target_speed = mechanism_config['min_target_speed_for_interceptor']
            if 'min_interceptor_distance' in mechanism_config:
                self.role_assigner.min_interceptor_distance = mechanism_config['min_interceptor_distance']
            if 'max_interceptor_distance' in mechanism_config:
                self.role_assigner.max_interceptor_distance = mechanism_config['max_interceptor_distance']
            if 'interceptor_prediction_steps' in mechanism_config:
                self.role_assigner.prediction_steps = mechanism_config['interceptor_prediction_steps']
            if 'interceptor_persistence_bonus' in mechanism_config:
                self.role_assigner.interceptor_persistence_bonus = mechanism_config['interceptor_persistence_bonus']
            if 'map_refresh_interval' in mechanism_config:
                self.map_refresh_interval = max(1, int(mechanism_config['map_refresh_interval']))
            if 'randomize_exit_zone' in mechanism_config:
                self.randomize_exit_zone = bool(mechanism_config['randomize_exit_zone'])
            if 'randomize_layout' in mechanism_config:
                self.randomize_layout = bool(mechanism_config['randomize_layout'])
            if 'intercept_projection_clearance' in mechanism_config:
                self.intercept_projection_clearance = float(mechanism_config['intercept_projection_clearance'])
            if 'target_obs_include_hunters' in mechanism_config:
                self.target_obs_include_hunters = bool(mechanism_config['target_obs_include_hunters'])
            if 'target_escape_sector_include_hunters' in mechanism_config:
                self.target_escape_sector_include_hunters = bool(mechanism_config['target_escape_sector_include_hunters'])
            if 'target_hunter_repulsion_scale' in mechanism_config:
                self.target_hunter_repulsion_scale = float(mechanism_config['target_hunter_repulsion_scale'])
            if 'target_ref_hunter_perception_range' in mechanism_config:
                self.ref_velocity_calculator.perception_range = float(mechanism_config['target_ref_hunter_perception_range'])
    
    def _collect_obs_info(self):
        multi_obs_info = []
        for obstacle in self.obstacles:
            multi_obs_info.append(obstacle._return_obs_info())

    def _build_target_assignment_weights(self, targets):
        if not targets:
            return {}
        diag = np.sqrt(2.0) * self.length
        weights = {}
        for target in targets:
            dist_to_exit = np.linalg.norm(target.position[:2] - self.escape_zone_center)
            escape_urgency = max(0.0, 1.0 - dist_to_exit / max(diag, 1e-6))
            weights[id(target)] = 1.0 + self.assignment_escape_pressure_coeff * escape_urgency
        return weights

    @staticmethod
    def _safe_normalize(vector, eps=1e-6):
        norm = float(np.linalg.norm(vector))
        if norm <= eps:
            return np.zeros(2, dtype=float)
        return np.asarray(vector, dtype=float) / norm

    def _get_escape_direction(self, target):
        for candidate in (
            getattr(target, 'reference_velocity', np.zeros(2, dtype=float)),
            target.velocity[:2],
            self.escape_zone_center - target.position[:2],
        ):
            direction = self._safe_normalize(candidate)
            if np.linalg.norm(direction) > 1e-6:
                return direction
        return np.array([1.0, 0.0], dtype=float)

    def _compute_target_escape_bandwidth(self, target, hunters):
        hunter_positions = [tuple(h.position[:2]) for h in hunters]
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
            histogram,
            angles,
            self.L_sensor,
        )
        if largest_escape_interval is None:
            return 0.0

        start_deg, end_deg = largest_escape_interval
        start = self._wrap_angle(np.deg2rad(start_deg))
        end = self._wrap_angle(np.deg2rad(end_deg))
        width = (end - start) % (2 * np.pi)
        if width < 1e-6:
            width = 2 * np.pi
        return float(np.clip(width / (2 * np.pi), 0.0, 1.0))

    def _update_escape_bandwidth_cache(self):
        target_hunter_groups = {}
        for hunter in self.hunters:
            if hunter.assigned_target is not None and self._is_target_active(hunter.assigned_target):
                target_hunter_groups.setdefault(hunter.assigned_target, []).append(hunter)
        self._prev_target_escape_bandwidths = {
            id(target): self._compute_target_escape_bandwidth(
                target,
                target_hunter_groups.get(target, []),
            )
            for target in self._get_active_targets()
        }

    def _project_intercept_point(self, point_xy, target):
        clearance = max(float(self.intercept_projection_clearance), 1e-3)
        point = np.asarray(point_xy, dtype=float).copy()
        point = np.clip(point, clearance, self.length - clearance)
        preferred = self._safe_normalize(point - target.position[:2])
        if np.linalg.norm(preferred) <= 1e-6:
            preferred = self._safe_normalize(target.velocity[:2])
        if np.linalg.norm(preferred) <= 1e-6:
            preferred = np.array([1.0, 0.0], dtype=float)

        for _ in range(8):
            adjusted = False
            point = np.clip(point, clearance, self.length - clearance)
            for obstacle in self.obstacles:
                delta = point - obstacle.position[:2]
                min_clearance = obstacle.radius + clearance
                distance = float(np.linalg.norm(delta))
                if distance >= min_clearance:
                    continue

                adjusted = True
                push_dir = self._safe_normalize(delta)
                if np.linalg.norm(push_dir) <= 1e-6:
                    push_dir = preferred
                blended = self._safe_normalize(0.75 * push_dir + 0.25 * preferred)
                if np.linalg.norm(blended) <= 1e-6:
                    blended = preferred
                point = obstacle.position[:2] + blended * min_clearance
                break
            if not adjusted:
                break

        return np.clip(point, clearance, self.length - clearance)

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _point_to_rect_distance(point, rect_bounds):
        x_min, x_max, y_min, y_max = rect_bounds
        dx = max(x_min - point[0], 0.0, point[0] - x_max)
        dy = max(y_min - point[1], 0.0, point[1] - y_max)
        return float(np.hypot(dx, dy))

    def _get_hunter_spawn_bounds(self):
        margin = self.spawn_clearance_margin
        return (
            max(0.0, self.hunter_spawn_low - margin),
            min(self.length, self.hunter_spawn_high + margin),
            max(0.0, self.hunter_spawn_low - margin),
            min(self.length, self.hunter_spawn_high + margin),
        )

    def _get_target_spawn_bounds(self):
        margin = self.spawn_clearance_margin
        return (
            max(0.0, self.target_spawn_low - margin),
            min(self.length, self.target_spawn_high + margin),
            max(0.0, self.target_spawn_low - margin),
            min(self.length, self.target_spawn_high + margin),
        )

    def _sample_exit_zone_center(self):
        margin = 0.15
        half = 0.5 * self.length
        candidates = [
            np.array([margin, margin], dtype=float),
            np.array([margin, self.length - margin], dtype=float),
            np.array([self.length - margin, margin], dtype=float),
            np.array([self.length - margin, self.length - margin], dtype=float),
            np.array([margin, half], dtype=float),
            np.array([self.length - margin, half], dtype=float),
            np.array([half, margin], dtype=float),
            np.array([half, self.length - margin], dtype=float),
        ]
        for idx in np.random.permutation(len(candidates)):
            candidate = candidates[idx]
            if self._point_to_rect_distance(candidate, self._get_hunter_spawn_bounds()) < self.escape_zone_radius:
                continue
            if self._point_to_rect_distance(candidate, self._get_target_spawn_bounds()) < self.escape_zone_radius:
                continue
            return candidate
        return candidates[0]

    def _is_obstacle_layout_valid(self, position_xy, radius, placed_obstacles, exit_center):
        for placed in placed_obstacles:
            distance = np.linalg.norm(position_xy - placed.position[:2])
            if distance < radius + placed.radius + self.layout_clearance_margin:
                return False

        if self._point_to_rect_distance(position_xy, self._get_hunter_spawn_bounds()) < radius:
            return False
        if self._point_to_rect_distance(position_xy, self._get_target_spawn_bounds()) < radius:
            return False
        if np.linalg.norm(position_xy - exit_center) < radius + self.escape_zone_radius + self.exit_clearance_margin:
            return False
        return True

    def _refresh_layout(self):
        if self.randomize_exit_zone:
            self.escape_zone_center = self._sample_exit_zone_center()

        placed_obstacles = []
        for obstacle in self.obstacles:
            placed = False
            for _ in range(200):
                candidate_pos = np.random.uniform(low=0.35, high=self.length - 0.35, size=(3,))
                candidate_pos[-1] = 0.0
                candidate_radius = np.random.uniform(0.1, 0.15)
                candidate_height = np.random.uniform(0.1, 0.15)
                if self._is_obstacle_layout_valid(candidate_pos[:2], candidate_radius, placed_obstacles, self.escape_zone_center):
                    obstacle.position = candidate_pos
                    obstacle.radius = candidate_radius
                    obstacle.height = candidate_height
                    placed_obstacles.append(obstacle)
                    placed = True
                    break
            if not placed:
                obstacle.position = np.array([0.3 + 0.2 * len(placed_obstacles), 1.0, 0.0], dtype=float)
                obstacle.radius = 0.1
                obstacle.height = 0.12
                placed_obstacles.append(obstacle)

        for agent in self.hunters + self.targets:
            agent.lidar.obstacles = self.obstacles

    def _count_targets_in_state(self, state):
        return sum(1 for target in self.targets if self._get_target_state(target) == state)

    def _get_outcome_code(self, timed_out=False):
        all_targets_captured = all(
            self._get_target_state(target) == 'captured' for target in self.targets
        )
        all_targets_resolved = all(
            self._get_target_state(target) in ('captured', 'escaped') for target in self.targets
        )
        any_target_escaped = any(
            self._get_target_state(target) == 'escaped' for target in self.targets
        )
        captured_target_count = self._count_targets_in_state('captured')

        if all_targets_captured:
            return 2
        if all_targets_resolved and any_target_escaped:
            return -1
        if timed_out:
            if any_target_escaped:
                return -1
            return 1 if captured_target_count > 0 else 0
        return None

    def _apply_outcome_bonus(self, rewards, outcome_code):
        if outcome_code is None:
            return 0.0, 0.0

        captured_ratio = self._count_targets_in_state('captured') / max(float(self.num_targets), 1.0)
        hunter_bonus = 0.0
        target_bonus = 0.0
        if outcome_code == 2:
            hunter_bonus = self.full_capture_outcome_reward
            target_bonus = self.target_full_capture_outcome_reward
        elif outcome_code == 1:
            hunter_bonus = self.partial_capture_outcome_reward * captured_ratio
            target_bonus = self.target_partial_capture_outcome_reward * captured_ratio
        elif outcome_code == 0:
            hunter_bonus = self.timeout_outcome_reward
            target_bonus = self.target_timeout_outcome_reward
        elif outcome_code == -1:
            hunter_bonus = self.escape_outcome_reward
            target_bonus = self.target_escape_outcome_reward

        if abs(hunter_bonus) > 1e-12:
            for i in range(self.num_hunters):
                rewards[i] += hunter_bonus
        if abs(target_bonus) > 1e-12:
            for i in range(self.num_targets):
                rewards[self.num_hunters + i] += target_bonus

        return hunter_bonus, target_bonus

    def finalize_timeout_outcome(self, rewards):
        outcome_code = self._get_outcome_code(timed_out=True)
        hunter_bonus, target_bonus = self._apply_outcome_bonus(rewards, outcome_code)
        return rewards, {
            'outcome_code': outcome_code,
            'captured_target_count': self._count_targets_in_state('captured'),
            'escaped_target_count': self._count_targets_in_state('escaped'),
            'all_targets_captured': outcome_code == 2,
            'episode_terminal': True,
            'capture_happened': outcome_code == 2,
            'escape_happened': outcome_code == -1,
            'outcome_hunter_bonus': hunter_bonus,
            'outcome_target_bonus': target_bonus,
        }

    def _compute_hunter_chase_reward(self, hunter, target):
        """
        Role-aware objective progress reward gated by heading quality.

        Chasers optimize progress to the target's current position; interceptors optimize
        progress to the projected interception point stored in hunter.target_position.
        """
        hunter_pos = hunter.position[:2]
        objective_pos = (
            hunter.target_position[:2]
            if hunter.assigned_target is not None
            else target.position[:2]
        )
        prev_hunter_pos = self._prev_hunter_positions.get(id(hunter), hunter_pos)
        prev_objective_pos = self._prev_hunter_objectives.get(id(hunter), objective_pos)

        d_prev = np.linalg.norm(prev_objective_pos - prev_hunter_pos)
        d_curr = np.linalg.norm(objective_pos - hunter_pos)
        progress = np.clip(d_prev - d_curr, -self.chase_progress_clip, self.chase_progress_clip)

        hunter_dir = hunter.velocity[:2]
        hunter_dir_norm = np.linalg.norm(hunter_dir)
        target_dir = objective_pos - hunter_pos
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

    def _compute_hunter_blocked_chase_terms(self, hunter, target, progress):
        if len(hunter.lasers) == 0:
            return 0.0, 0.0, 0.0

        objective_pos = (
            hunter.target_position[:2]
            if hunter.assigned_target is not None
            else target.position[:2]
        )
        target_vector = objective_pos - hunter.position[:2]
        target_distance = np.linalg.norm(target_vector)
        if target_distance <= 1e-6:
            return 0.0, 0.0, 0.0

        max_range = max(float(hunter.lidar.max_detect_d), 1e-6)
        laser_angles = np.asarray(hunter.lidar.angles, dtype=float)
        laser_distances = np.asarray(hunter.lasers, dtype=float)
        target_angle = float(np.arctan2(target_vector[1], target_vector[0]))
        angle_diffs = np.array([self._wrap_to_pi(angle - target_angle) for angle in laser_angles], dtype=float)
        direct_idx = int(np.argmin(np.abs(angle_diffs)))
        direct_clearance = float(np.clip(laser_distances[direct_idx], 0.0, max_range))
        visibility_limit = min(target_distance, max_range)
        blocked_ratio = float(np.clip((visibility_limit - direct_clearance) / max(visibility_limit, 1e-6), 0.0, 1.0))

        if blocked_ratio <= 1e-6:
            return 0.0, 0.0, 0.0

        open_threshold = 0.72 * max_range
        feasible_mask = laser_distances >= open_threshold
        if np.any(feasible_mask):
            feasible_indices = np.where(feasible_mask)[0]
            best_idx = int(feasible_indices[np.argmin(np.abs(angle_diffs[feasible_indices]))])
        else:
            score = laser_distances / max_range - 0.12 * np.abs(angle_diffs) / np.pi
            best_idx = int(np.argmax(score))

        gap_angle = float(laser_angles[best_idx])
        gap_direction = np.array([np.cos(gap_angle), np.sin(gap_angle)], dtype=float)
        gap_clearance = float(np.clip(laser_distances[best_idx] / max_range, 0.0, 1.0))

        hunter_velocity = hunter.velocity[:2]
        velocity_norm = np.linalg.norm(hunter_velocity)
        if velocity_norm > 1e-6:
            gap_alignment = float(np.dot(hunter_velocity, gap_direction) / velocity_norm)
        else:
            gap_alignment = 0.0

        progress_ratio = float(np.clip(progress / max(self.chase_progress_clip, 1e-6), -1.0, 1.0))
        gap_reward = blocked_ratio * gap_clearance * max(gap_alignment, 0.0) * (0.5 + 0.5 * max(progress_ratio, 0.0))

        speed_ratio = float(np.clip(velocity_norm / max(self.v_max, 1e-6), 0.0, 1.0))
        stalled_ratio = max(0.0, 0.35 - speed_ratio) / 0.35
        lack_of_progress = 1.0 - max(progress_ratio, 0.0)
        stuck_penalty = blocked_ratio * stalled_ratio * lack_of_progress

        return blocked_ratio, gap_reward, stuck_penalty

    def _compute_chaser_slot_rewards(self, target, chasers):
        if not chasers:
            return {}, 0.0

        target_pos = target.position[:2]
        escape_dir = self._get_escape_direction(target)
        flank_dir = np.array([-escape_dir[1], escape_dir[0]], dtype=float)
        forward_offset = float(self.containment_slot_forward_offset)
        lateral_offset = float(self.containment_slot_lateral_offset)
        slot_sigma = max(lateral_offset, 1e-3)
        slots = [
            target_pos + forward_offset * escape_dir + lateral_offset * flank_dir,
            target_pos + forward_offset * escape_dir - lateral_offset * flank_dir,
        ]

        slot_rewards = {}
        signed_side_scores = []
        for hunter in chasers:
            hunter_pos = hunter.position[:2]
            min_slot_distance = min(np.linalg.norm(slot - hunter_pos) for slot in slots)
            slot_rewards[id(hunter)] = float(
                np.exp(-(min_slot_distance ** 2) / max(2.0 * slot_sigma ** 2, 1e-6))
            )

            rel = hunter_pos - target_pos
            lateral = float(np.dot(rel, flank_dir))
            forward = float(np.dot(rel, escape_dir))
            forward_gate = float(
                np.exp(-((forward - forward_offset) ** 2) / max(2.0 * forward_offset ** 2, 1e-6))
            )
            if abs(lateral) > 1e-6:
                signed_side_scores.append(np.sign(lateral) * forward_gate)

        side_balance = 0.0
        if len(signed_side_scores) >= 2:
            side_balance = float(np.clip(1.0 - abs(np.mean(signed_side_scores)), 0.0, 1.0))

        return slot_rewards, side_balance

    def _compute_interceptor_quality_reward(self, hunter, target):
        intercept_pos = hunter.target_position[:2]
        target_pos = target.position[:2]
        target_to_intercept = float(np.linalg.norm(intercept_pos - target_pos))
        hunter_to_intercept = float(np.linalg.norm(intercept_pos - hunter.position[:2]))
        if target_to_intercept <= 1e-6:
            return 0.0

        escape_dir = self._get_escape_direction(target)
        forward_progress = float(np.dot(hunter.position[:2] - target_pos, escape_dir))
        forward_gate = float(np.clip(0.5 + 0.5 * np.tanh(forward_progress / 0.06), 0.0, 1.0))
        arrival_margin = (target_to_intercept - hunter_to_intercept) / max(
            target_to_intercept + hunter_to_intercept,
            1e-6,
        )
        return float(np.clip(arrival_margin, -1.0, 1.0) * forward_gate)

    def _compute_target_alignment_gate(self, target):
        wall_threshold = 0.6 * self.L_sensor
        min_wall_distance = min(
            target.position[0],
            target.position[1],
            self.length - target.position[0],
            self.length - target.position[1],
        )
        wall_gate = float(np.clip(min_wall_distance / max(wall_threshold, 1e-6), 0.0, 1.0))
        if self._target_boundary_clipped.get(id(target), False):
            wall_gate *= 0.2
        return wall_gate

    def _set_target_state(self, target, state):
        self.target_states[id(target)] = state

    def _get_target_state(self, target):
        return self.target_states.get(id(target), 'active')

    def _is_target_active(self, target):
        return self._get_target_state(target) == 'active'

    def _get_active_targets(self):
        return [target for target in self.targets if self._is_target_active(target)]

    def reset(self):
        '''
        Reset the environment to an initial state and returns the initial observations.
        Returns:
            h_obs (list of np.array): Observations for all hunters.
            t_obs (list of np.array): Observations for all targets.
        '''
        if self.randomize_layout:
            refresh_interval = max(1, int(self.map_refresh_interval))
            if self._reset_count == 0 or self._reset_count % refresh_interval == 0:
                self._refresh_layout()
        self._reset_count += 1

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
                hunter.position = np.random.uniform(low=self.hunter_spawn_low, high=self.hunter_spawn_high, size=(3,))
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
                target.position = np.random.uniform(low=self.target_spawn_low, high=self.target_spawn_high, size=(3,))
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
            self._set_target_state(target, 'active')

        # Reset role assigner (clear Kalman filters)
        self.role_assigner.reset()
        
        # Initialize Kalman filters for all targets
        for target in self.targets:
            target_id = id(target)
            self.role_assigner.update_kalman_filter(
                target_id,
                target.position[:2],
                velocity=target.velocity[:2],
            )

        # Assign initial targets to hunters
        self._assign_targets_to_hunters()
        
        # Assign initial roles to hunters
        self._assign_hunter_roles()

        for target in self.targets:
            if not self._is_target_active(target):
                target.reference_velocity = np.zeros(2)
            elif self.use_ref_velocity:
                target.reference_velocity = self.ref_velocity_calculator.compute_reference_velocity(
                    target,
                    self.hunters,
                    escape_zone_center=self.escape_zone_center,
                    hunter_weight_scale=self.target_hunter_repulsion_scale,
                )
            else:
                target.reference_velocity = np.zeros(2)

        self._prev_hunter_positions = {
            id(hunter): hunter.position[:2].copy() for hunter in self.hunters
        }
        self._prev_target_positions = {
            id(target): target.position[:2].copy() for target in self.targets
        }
        self._prev_hunter_objectives = {
            id(hunter): hunter.target_position[:2].copy() for hunter in self.hunters
        }
        self._prev_target_ref_velocities = {
            id(target): target.reference_velocity.copy() for target in self.targets
        }
        self._target_boundary_clipped = {
            id(target): False for target in self.targets
        }
        self._update_escape_bandwidth_cache()

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
        self._prev_hunter_objectives = {
            id(hunter): hunter.target_position[:2].copy() for hunter in self.hunters
        }
        self._prev_target_ref_velocities = {
            id(target): target.reference_velocity.copy() for target in self.targets
        }
        self._target_boundary_clipped = {
            id(target): False for target in self.targets
        }

        # Apply actions to hunters
        for i, hunter in enumerate(self.hunters):
            hunter.move(actions[i], self.v_max)
        
        # Apply actions to targets
        for i, target in enumerate(self.targets):
            if self._is_target_active(target):
                target.move(actions[self.num_hunters + i], self.v_max) # Assume "action" set is combination of hunetrs' & targets'
            else:
                target.velocity = np.zeros(3)

        # Clip to boundaries before scanning so the new reference velocity matches
        # the actual post-transition state seen in next observations.
        for hunter in self.hunters:
            hunter.position[:2] = np.clip(hunter.position[:2], 0, self.length)
        for target in self.targets:
            unclipped = target.position[:2].copy()
            clipped = np.clip(unclipped, 0, self.length)
            self._target_boundary_clipped[id(target)] = bool(np.any(np.abs(unclipped - clipped) > 1e-9))
            target.position[:2] = clipped

        # Update lasers after movement
        for hunter in self.hunters:
            hunter.lasers = hunter.lidar.scan(hunter.position, self.length)
        for target in self.targets:
            target.lasers = target.lidar.scan(target.position, self.length)

        # Update reference velocity for each target
        for target in self.targets:
            if not self._is_target_active(target):
                target.reference_velocity = np.zeros(2)
            elif self.use_ref_velocity:
                # 完整引导速度：逃逸向量 + 避障向量 + 出口吸引力
                target.reference_velocity = self.ref_velocity_calculator.compute_reference_velocity(
                    target,
                    self.hunters,
                    escape_zone_center=self.escape_zone_center,
                    hunter_weight_scale=self.target_hunter_repulsion_scale,
                )
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

        if not self.use_ref_velocity:
            for target in self.targets:
                if self._is_target_active(target):
                    target.reference_velocity = self.ref_velocity_calculator.compute_reference_velocity(
                        target,
                        [],
                        escape_zone_center=self.escape_zone_center,
                        hunter_weight_scale=0.0,
                    )

        # Assign targets to hunters using density field allocator
        self._assign_targets_to_hunters()
        
        # Update Kalman filters for all targets
        for target in self.targets:
            if self._is_target_active(target):
                target_id = id(target)
                self.role_assigner.update_kalman_filter(
                    target_id,
                    target.position[:2],
                    velocity=target.velocity[:2],
                )
        
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
        active_targets = self._get_active_targets()
        if not active_targets:
            assignments = {}
        elif self.use_density_field:
            target_weights = self._build_target_assignment_weights(active_targets)
            # 使用密度场分配器计算最优分配
            assignments = self.density_allocator.assign_targets(
                self.hunters,
                active_targets,
                self.obstacles,
                target_weights=target_weights,
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
        target_hunter_groups = {}
        for hunter in self.hunters:
            if hunter.assigned_target is not None:
                target_hunter_groups.setdefault(hunter.assigned_target, []).append(hunter)

        for hunter in self.hunters:
            if hunter.assigned_target is None:
                hunter.assigned_group_size = 0
            else:
                hunter.assigned_group_size = len(target_hunter_groups.get(hunter.assigned_target, []))

        if not self.use_role_assignment:
            # 消融模式：所有hunter均为chaser
            for hunter in self.hunters:
                hunter.role = 'chaser'
                if hunter.assigned_target is not None:
                    hunter.target_position = hunter.assigned_target.position.copy()
                else:
                    hunter.target_position = np.zeros(3)
            return
        
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
                        hunter,
                        target,
                        hunter.role,
                        project_fn=self._project_intercept_point,
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
        for target in self._get_active_targets():
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
                nearest_hunters_xy = hunter_xy[nearest_indices]
            else:
                other_hunters = np.delete(hunter_xy, i, axis=0)
                # If less than two other hunters, pad with zeros
                nearest_hunters_xy = np.zeros((2, 2), dtype=float)
                if len(other_hunters) == 1:
                    nearest_hunters_xy[0] = other_hunters[0]
                else:
                    nearest_hunters_xy = np.zeros((2, 2), dtype=float)

            # Get velocity
            velocity_xy = hunter.velocity[:2]
            own_pos_xy = hunter.position[:2]
            nearest_rel_xy = nearest_hunters_xy - own_pos_xy

            if hunter.assigned_target is not None:
                target_delta_xy = hunter.assigned_target.position[:2] - own_pos_xy
                target_velocity_xy = hunter.assigned_target.velocity[:2]
            else:
                target_delta_xy = np.zeros(2, dtype=float)
                target_velocity_xy = np.zeros(2, dtype=float)

            pursuit_delta_xy = hunter.target_position[:2] - own_pos_xy
            role_flag = 1.0 if hunter.role == 'interceptor' else 0.0
            group_size_norm = hunter.assigned_group_size / max(self.num_hunters, 1)

            # Get laser data
            laser_data = hunter.lasers  # Assuming it's a 1D array of size num_lasers

            # Concatenate all observation components
            obs = np.concatenate([
                nearest_rel_xy.flatten() / self.length,                  # 2 * 2 = 4
                own_pos_xy / self.length,                                # 2
                velocity_xy / self.v_max,                                # 2
                target_delta_xy / self.length,                           # 2
                target_velocity_xy / self.v_max,                         # 2
                pursuit_delta_xy / self.length,                          # 2
                np.array([role_flag, group_size_norm], dtype=float),     # 2
                laser_data / self.L_sensor                               # 16
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
            if not self.target_obs_include_hunters:
                nearest_hunters = np.zeros((3, 3), dtype=float)

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

        if self.target_escape_sector_include_hunters:
            hunter_positions = [tuple(h.position[:2]) for h in self.hunters]
        else:
            hunter_positions = []
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
        gap_progress_rewards = [0.0] * self.num_hunters
        stuck_penalties = [0.0] * self.num_hunters
        containment_rewards = [0.0] * self.num_hunters
        role_geometry_rewards = [0.0] * self.num_hunters
        blocked_chase_ratios = []
        current_escape_bandwidths = {}

        # Pre-compute index mappings to avoid repeated .index() calls
        hunter_index_map = {id(h): i for i, h in enumerate(self.hunters)}
        target_index_map = {id(t): i for i, t in enumerate(self.targets)}
        for target in self.targets:
            if not self._is_target_active(target):
                dones[self.num_hunters + target_index_map[id(target)]] = True

        # Map each active target to its assigned hunters
        target_hunter_groups = {}
        for target in self._get_active_targets():
            target_hunter_groups[target] = [hunter for hunter in self.hunters if hunter.assigned_target == target]
        group_sizes = [len(hunters) for hunters in target_hunter_groups.values()]
        active_target_count_before = len(target_hunter_groups)
        min_group_size = min(group_sizes) if group_sizes else 0
        max_group_size = max(group_sizes) if group_sizes else 0
        group_size_std = float(np.std(group_sizes)) if group_sizes else 0.0
        interceptor_count = sum(
            1 for hunter in self.hunters
            if hunter.assigned_target is not None and hunter.role == 'interceptor'
        )

        capture_happened = False
        captured_target_indices = []
        escape_happened = False
        escaped_target_indices = []

        # 检查逃逸区域
        for target in self.targets:
            if not self._is_target_active(target):
                continue
            target_index = target_index_map[id(target)]
            dist_to_escape = np.linalg.norm(target.position[:2] - self.escape_zone_center)
            if dist_to_escape <= self.escape_zone_radius:
                # Target 到达逃逸区域，逃逸成功
                self._set_target_state(target, 'escaped')
                escape_happened = True
                escaped_target_indices.append(target_index)
                rewards[self.num_hunters + target_index] += self.escape_reward_for_target
                dones[self.num_hunters + target_index] = True
                # 所有猎手受惩罚
                for i in range(self.num_hunters):
                    rewards[i] += self.escape_penalty_for_hunters

        # Reward for hunters chasing and capturing targets
        for target, hunters in target_hunter_groups.items():
            current_escape_bandwidth = self._compute_target_escape_bandwidth(target, hunters)
            current_escape_bandwidths[id(target)] = current_escape_bandwidth
            prev_escape_bandwidth = self._prev_target_escape_bandwidths.get(
                id(target),
                current_escape_bandwidth,
            )
            containment_progress = float(
                np.clip(prev_escape_bandwidth - current_escape_bandwidth, -0.2, 0.2)
            )
            containment_quality = float(np.clip(1.0 - current_escape_bandwidth, 0.0, 1.0))
            chasers = [hunter for hunter in hunters if hunter.role != 'interceptor']
            slot_rewards, side_balance = self._compute_chaser_slot_rewards(target, chasers)
            side_balance_share = side_balance / max(len(chasers), 1)

            # calculate chasing reward and ifrounded reward
            for hunter in hunters:
                chase_reward, progress, _ = self._compute_hunter_chase_reward(hunter, target)
                blocked_ratio, gap_reward, stuck_penalty = self._compute_hunter_blocked_chase_terms(
                    hunter, target, progress
                )
                hunter_index = hunter_index_map[id(hunter)]
                rewards[hunter_index] += self.chase_reward_coeff * chase_reward
                chase_rewards[hunter_index] += self.chase_reward_coeff * chase_reward
                rewards[hunter_index] += self.blocked_chase_reward_coeff * gap_reward
                gap_progress_rewards[hunter_index] += self.blocked_chase_reward_coeff * gap_reward
                rewards[hunter_index] -= self.stuck_penalty_coeff * stuck_penalty
                stuck_penalties[hunter_index] += self.stuck_penalty_coeff * stuck_penalty
                blocked_chase_ratios.append(blocked_ratio)

                containment_share = 1.0 if hunter.role == 'chaser' else 0.6
                containment_reward = (
                    self.containment_progress_reward_coeff * containment_share * containment_progress
                    + self.containment_quality_reward_coeff * containment_share * containment_quality
                )
                rewards[hunter_index] += containment_reward
                containment_rewards[hunter_index] += containment_reward

                if hunter.role == 'interceptor':
                    role_reward = (
                        self.interceptor_quality_reward_coeff
                        * self._compute_interceptor_quality_reward(hunter, target)
                    )
                else:
                    role_reward = (
                        self.chaser_slot_reward_coeff * slot_rewards.get(id(hunter), 0.0)
                        + self.chaser_side_balance_reward_coeff * side_balance_share
                    )
                rewards[hunter_index] += role_reward
                role_geometry_rewards[hunter_index] += role_reward

            multi_hunters_pos = [h.position for h in hunters]
            if utils.isRounded(tuple(target.position[:2]), [tuple(row[:2]) for row in multi_hunters_pos], self.L_sensor, self.max_escape_angle,
                               boundary_length=self.length, obstacles=self.obstacles):
                for hunter in hunters:
                    hunter_index = hunter_index_map[id(hunter)]
                    rewards[hunter_index] += self.capture_reward
                    capture_rewards[hunter_index] += self.capture_reward
                target_index = target_index_map[id(target)]
                self._set_target_state(target, 'captured')
                dones[self.num_hunters + target_index] = True
                captured_target_indices.append(target_index)

        if captured_target_indices:
            for i in range(self.num_hunters):
                rewards[i] += self.team_capture_bonus * len(captured_target_indices)

        # Reward for targets
        for target in self.targets:
            target_index = target_index_map[id(target)]
            if not self._is_target_active(target):
                continue

            escape_reward = self._compute_escape_sector_reward(target)
            rewards[self.num_hunters + target_index] += self.escape_reward_coeff * escape_reward
            escape_rewards[target_index] += self.escape_reward_coeff * escape_reward

            # Add reference velocity alignment reward
            # Use target.reference_velocity already computed in step() — avoid recomputation
            if self.use_ref_velocity:
                v_ref = self._prev_target_ref_velocities.get(id(target), target.reference_velocity)
            else:
                v_ref = np.zeros(2)
            v_actual = target.velocity[:2]
            alignment_gate = self._compute_target_alignment_gate(target)
            
            # Compute cosine similarity: cos(v_actual, v_ref) = (v_actual · v_ref) / (||v_actual|| * ||v_ref||)
            v_actual_norm = np.linalg.norm(v_actual)
            v_ref_norm = np.linalg.norm(v_ref)
            
            epsilon = 1e-6
            if v_actual_norm > epsilon and v_ref_norm > epsilon:
                cosine_similarity = np.dot(v_actual, v_ref) / (v_actual_norm * v_ref_norm)
                alignment_reward = alignment_gate * cosine_similarity
            else:
                # If either velocity is zero, no alignment reward
                alignment_reward = 0.0
            
            rewards[self.num_hunters + target_index] += self.alignment_reward_coeff * alignment_reward
            alignment_rewards[target_index] += self.alignment_reward_coeff * alignment_reward

            obstacle_warning_threshold = self.target_obstacle_warning_ratio * self.L_sensor
            min_laser_length = float(np.min(target.lasers)) if len(target.lasers) > 0 else self.L_sensor
            if min_laser_length < obstacle_warning_threshold:
                proximity_ratio = (obstacle_warning_threshold - min_laser_length) / max(obstacle_warning_threshold, 1e-6)
                rewards[self.num_hunters + target_index] -= (
                    self.obstacle_proximity_penalty_coeff * proximity_ratio
                )

            # Add obstacle interior penalty
            if self.ref_velocity_calculator.is_inside_obstacle(target.position, self.obstacles):
                rewards[self.num_hunters + target_index] -= self.obstacle_interior_penalty

        active_target_count_after = len(self._get_active_targets())
        if self.hunter_time_penalty_coeff > 0.0 and active_target_count_after > 0:
            time_penalty = self.hunter_time_penalty_coeff * active_target_count_after
            for i in range(self.num_hunters):
                rewards[i] -= time_penalty

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
            for target in self._get_active_targets():
                distance = np.linalg.norm(hunter.position[:2] - target.position[:2])
                if distance < self.distance_threshold:
                    rewards[self.num_hunters + target_index_map[id(target)]] -= (
                        self.target_hunter_contact_penalty_coeff
                        * self.safe_penalty_coeff
                        * (self.distance_threshold - distance)
                    )
        
        # between targets
        active_targets = self._get_active_targets()
        for i in range(len(active_targets)):
            for j in range(i+1, len(active_targets)):
                distance = np.linalg.norm(active_targets[i].position[:2] - active_targets[j].position[:2])
                if distance < self.distance_threshold:
                    penalty = self.safe_penalty_coeff * (self.distance_threshold - distance)
                    rewards[self.num_hunters + target_index_map[id(active_targets[i])]] -= penalty
                    rewards[self.num_hunters + target_index_map[id(active_targets[j])]] -= penalty

        # reward for safety (uav-obstacles)
        # Only penalize when very close to obstacles (< 50% of sensor range)
        obstacle_danger_threshold = self.L_sensor * self.obstacle_danger_ratio
        for agent in self.hunters + self._get_active_targets():
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

        captured_target_count = self._count_targets_in_state('captured')
        escaped_target_count = self._count_targets_in_state('escaped')
        outcome_code = self._get_outcome_code(timed_out=False)
        outcome_hunter_bonus = 0.0
        outcome_target_bonus = 0.0
        if outcome_code is not None:
            outcome_hunter_bonus, outcome_target_bonus = self._apply_outcome_bonus(rewards, outcome_code)

        reward_info = {
            'chase_rewards': chase_rewards,
            'capture_rewards': capture_rewards,
            'escape_rewards': escape_rewards,
            'alignment_rewards': alignment_rewards,
            'gap_progress_rewards': gap_progress_rewards,
            'stuck_penalties': stuck_penalties,
            'containment_rewards': containment_rewards,
            'role_geometry_rewards': role_geometry_rewards,
            'capture_happened': False,
            'captured_targets': captured_target_indices,
            'escape_happened': escape_happened,
            'escaped_targets': escaped_target_indices,
            'stage_name': self.current_stage_name,
            'active_target_count': active_target_count_after,
            'min_group_size': min_group_size,
            'max_group_size': max_group_size,
            'group_size_std': group_size_std,
            'interceptor_count': interceptor_count,
            'blocked_chase_ratio': float(np.mean(blocked_chase_ratios)) if blocked_chase_ratios else 0.0,
            'avg_gap_reward': float(np.mean(gap_progress_rewards)) if gap_progress_rewards else 0.0,
            'avg_stuck_penalty': float(np.mean(stuck_penalties)) if stuck_penalties else 0.0,
            'avg_containment_reward': float(np.mean(containment_rewards)) if containment_rewards else 0.0,
            'avg_role_geometry_reward': float(np.mean(role_geometry_rewards)) if role_geometry_rewards else 0.0,
            'avg_escape_bandwidth': float(np.mean(list(current_escape_bandwidths.values()))) if current_escape_bandwidths else 1.0,
            'captured_target_count': captured_target_count,
            'escaped_target_count': escaped_target_count,
            'outcome_code': outcome_code,
            'outcome_hunter_bonus': outcome_hunter_bonus,
            'outcome_target_bonus': outcome_target_bonus,
        }
        all_targets_captured = all(
            self._get_target_state(target) == 'captured' for target in self.targets
        )
        all_targets_resolved = all(
            self._get_target_state(target) in ('captured', 'escaped') for target in self.targets
        )
        capture_happened = all_targets_captured
        reward_info['capture_happened'] = capture_happened
        reward_info['all_targets_captured'] = all_targets_captured
        reward_info['episode_terminal'] = all_targets_captured or all_targets_resolved
        self._prev_target_escape_bandwidths = current_escape_bandwidths
        if reward_info['episode_terminal'] and self.capture_ends_episode:
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
        self.assigned_group_size = 0
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
