"""
密度场分配器 (Density Field Allocator)

基于流体力学类比的密度场目标分配机制，用于为hunter智能体分配追击目标。
密度场综合考虑智能体聚集效应、距离衰减、速度匹配和障碍物规避四个因素。
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


def safe_divide(numerator: float, denominator: float, epsilon: float = 1e-6) -> float:
    """
    安全除法，防止除零错误
    
    参数:
        numerator: 分子
        denominator: 分母
        epsilon: 小常数，防止除零
        
    返回:
        除法结果
    """
    return numerator / (denominator + epsilon)


class DensityFieldAllocator:
    """
    密度场分配器，实现基于密度场的目标分配机制
    
    密度场公式: ρ_j(t) = w_j × Φ_j(t) × V_j(t) × Ω_j(t)
    其中:
        - Φ_j: 智能体聚集效应
        - V_j: 速度匹配因子
        - Ω_j: 障碍物削弱因子
        - w_j: 目标权重
    """
    
    def __init__(self, h: float = 0.1, alpha: float = 0.5, beta: float = 0.3,
                 sigma: float = 0.05, delta: float = 1e-6,
                 underloaded_priority: float = 1.2,
                 over_assignment_penalty: float = 0.8,
                 density_value_weight: float = 1.0,
                 target_proximity_weight: float = 2.6,
                 slot_proximity_weight: float = 1.4,
                 switch_penalty: float = 0.45,
                 target_inertia_bonus: float = 0.35,
                 slot_inertia_bonus: float = 0.20,
                 target_distance_scale: float = 0.28,
                 slot_distance_scale: float = 0.18,
                 slot_radius: float = 0.12,
                 slot_clearance_margin: float = 0.04):
        """
        初始化密度场分配器
        
        参数:
            h: 核函数带宽，控制距离衰减速度
            alpha: 速度匹配权重系数
            beta: 障碍物削弱强度 (0 < beta < 1)
            sigma: 障碍物影响范围
            delta: 效用计算中的小常数，防止除零
        """
        self.h = h
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.delta = delta
        self.underloaded_priority = underloaded_priority
        self.over_assignment_penalty = over_assignment_penalty
        self.density_value_weight = density_value_weight
        self.target_proximity_weight = target_proximity_weight
        self.slot_proximity_weight = slot_proximity_weight
        self.switch_penalty = switch_penalty
        self.target_inertia_bonus = target_inertia_bonus
        self.slot_inertia_bonus = slot_inertia_bonus
        self.target_distance_scale = target_distance_scale
        self.slot_distance_scale = slot_distance_scale
        self.slot_radius = slot_radius
        self.slot_clearance_margin = slot_clearance_margin
    
    def gaussian_kernel(self, distance: float) -> float:
        """
        高斯核函数 K(d/h) = exp(-0.5 * (d/h)^2)
        
        参数:
            distance: 距离值
            
        返回:
            核函数值，范围 (0, 1]
        """
        # 防止数值溢出，限制最大距离
        distance = np.clip(distance, 0, 10.0)
        return np.exp(-0.5 * (distance / self.h) ** 2)
    
    def compute_aggregation_effect(self, target_pos: np.ndarray, 
                                   hunter_positions: List[np.ndarray]) -> float:
        """
        计算智能体聚集效应 Φ_j(t)
        
        Φ_j(t) = Σ_{i=1}^{M} K(||x_target_j - x_hunter_i|| / h)
        
        参数:
            target_pos: 目标位置 [x, y, z]
            hunter_positions: hunter位置列表
            
        返回:
            聚集效应值
        """
        if len(hunter_positions) == 0:
            return 0.0
        
        aggregation = 0.0
        for hunter_pos in hunter_positions:
            # 计算2D距离（忽略z轴）
            distance = np.linalg.norm(target_pos[:2] - hunter_pos[:2])
            aggregation += self.gaussian_kernel(distance)
        
        return aggregation
    
    def compute_velocity_matching(self, target_pos: np.ndarray,
                                  hunters: List) -> float:
        """
        计算速度匹配因子 V_j(t)
        
        V_j(t) = 1 + α * Σ_{i=1}^{M} [(v_hunter_i · (x_target_j - x_hunter_i)) / 
                                       (||v_hunter_i|| * ||x_target_j - x_hunter_i|| + ε)] * 
                                       K(||x_target_j - x_hunter_i|| / h)
        
        参数:
            target_pos: 目标位置 [x, y, z]
            hunters: Hunter对象列表
            
        返回:
            速度匹配因子
        """
        if len(hunters) == 0:
            return 1.0
        
        velocity_factor = 0.0
        epsilon = 1e-6
        
        for hunter in hunters:
            # 计算方向向量（2D）
            direction = target_pos[:2] - hunter.position[:2]
            distance = np.linalg.norm(direction)
            
            # hunter速度（2D）
            hunter_vel = hunter.velocity[:2]
            vel_norm = np.linalg.norm(hunter_vel)
            
            # 计算速度与方向的对齐程度
            if vel_norm > epsilon and distance > epsilon:
                alignment = np.dot(hunter_vel, direction) / (vel_norm * distance + epsilon)
                kernel_value = self.gaussian_kernel(distance)
                velocity_factor += alignment * kernel_value
        
        return 1.0 + self.alpha * velocity_factor

    
    def compute_obstacle_attenuation(self, target_pos: np.ndarray,
                                    obstacles: List) -> float:
        """
        计算障碍物削弱因子 Ω_j(t)
        
        Ω_j(t) = Π_{k} [1 - β * exp(-||x_target_j - x_obstacle_k||^2 / σ^2)]
        
        参数:
            target_pos: 目标位置 [x, y, z]
            obstacles: Obstacle对象列表
            
        返回:
            障碍物削弱因子，范围 (0, 1]
        """
        if len(obstacles) == 0:
            return 1.0
        
        attenuation = 1.0
        
        for obstacle in obstacles:
            # 计算到障碍物中心的距离（2D）
            distance = np.linalg.norm(target_pos[:2] - obstacle.position[:2])
            
            # 计算削弱因子
            factor = 1.0 - self.beta * np.exp(-(distance ** 2) / (self.sigma ** 2))
            attenuation *= factor
        
        return attenuation
    
    def compute_density_field(self, target, hunters: List,
                             obstacles: List, w_j: float = 1.0) -> float:
        """
        计算目标j的密度场 ρ_j(t) = w_j * Φ_j * V_j * Ω_j
        
        参数:
            target: Target对象
            hunters: Hunter对象列表
            obstacles: Obstacle对象列表
            w_j: 目标权重
            
        返回:
            密度场值
        """
        target_pos = target.position
        hunter_positions = [h.position for h in hunters]
        
        # 计算四个因子
        phi_j = self.compute_aggregation_effect(target_pos, hunter_positions)
        v_j = self.compute_velocity_matching(target_pos, hunters)
        omega_j = self.compute_obstacle_attenuation(target_pos, obstacles)
        
        # 密度场 = 权重 × 聚集效应 × 速度匹配 × 障碍物削弱
        density = w_j * phi_j * v_j * omega_j
        
        return density
    
    def compute_marginal_contribution(self, hunter, target,
                                     other_hunters: List, obstacles: List, 
                                     w_j: float = 1.0) -> float:
        """
        计算hunter对target的边际贡献 Δρ_{j←i}
        
        边际贡献 = ρ_j(包含该hunter) - ρ_j(不包含该hunter)
        
        参数:
            hunter: Hunter对象
            target: Target对象
            other_hunters: 其他Hunter对象列表（不包含当前hunter）
            obstacles: Obstacle对象列表
            w_j: 目标权重
            
        返回:
            边际贡献值
        """
        # 计算包含该hunter的密度场
        all_hunters = [hunter] + other_hunters
        density_with = self.compute_density_field(target, all_hunters, obstacles, w_j)
        
        # 计算不包含该hunter的密度场
        density_without = self.compute_density_field(target, other_hunters, obstacles, w_j)
        
        # 边际贡献
        marginal = density_with - density_without
        
        return max(0.0, marginal)  # 确保非负
    
    def compute_utility(self, hunter, target,
                       other_hunters: List, obstacles: List,
                       w_j: float = 1.0) -> float:
        """
        计算追击效用 U_{ij} = Δρ_{j←i} / (ρ_j + δ)
        
        参数:
            hunter: Hunter对象
            target: Target对象
            other_hunters: 其他Hunter对象列表（不包含当前hunter）
            obstacles: Obstacle对象列表
            w_j: 目标权重
            
        返回:
            追击效用值
        """
        # 计算边际贡献
        marginal = self.compute_marginal_contribution(hunter, target, other_hunters, obstacles, w_j)
        
        # 计算当前密度场
        all_hunters = [hunter] + other_hunters
        density = self.compute_density_field(target, all_hunters, obstacles, w_j)
        
        # 计算效用，使用安全除法防止除零
        utility = safe_divide(marginal, density, self.delta)
        
        return utility
    
    def _compute_desired_coverages(self, num_hunters: int, num_targets: int) -> List[int]:
        """
        Compute a balanced desired coverage plan for active targets.
        """
        if num_targets <= 0:
            return []
        base = num_hunters // num_targets
        remainder = num_hunters % num_targets
        desired = [base] * num_targets
        for i in range(remainder):
            desired[i] += 1
        return desired

    def _coverage_factor(self, current_count: int, desired_count: int) -> float:
        """
        Prefer under-covered targets and damp over-saturated ones.
        """
        if desired_count <= 0:
            return 1.0 / (1.0 + self.over_assignment_penalty * max(current_count, 0))
        if current_count < desired_count:
            gap_ratio = (desired_count - current_count) / desired_count
            return 1.0 + self.underloaded_priority * gap_ratio
        overload = current_count - desired_count
        return 1.0 / (1.0 + self.over_assignment_penalty * overload)

    def _target_weight(self, target, target_weights: Optional[Dict[int, float]]) -> float:
        if not target_weights:
            return 1.0
        return float(target_weights.get(id(target), 1.0))

    @staticmethod
    def _safe_normalize(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= eps:
            return np.zeros(2, dtype=float)
        return np.asarray(vector, dtype=float) / norm

    def _build_slot_angles(self, desired_count: int) -> List[float]:
        if desired_count <= 1:
            return [0.0]
        if desired_count == 2:
            return list(np.deg2rad([-35.0, 35.0]))
        if desired_count == 3:
            return list(np.deg2rad([-65.0, 0.0, 65.0]))
        if desired_count == 4:
            return list(np.deg2rad([-95.0, -30.0, 30.0, 95.0]))
        return list(np.deg2rad(np.linspace(-110.0, 110.0, desired_count)))

    def _project_slot_position(
        self,
        point_xy: np.ndarray,
        target_pos_xy: np.ndarray,
        obstacles: List,
        boundary_length: Optional[float] = None,
    ) -> np.ndarray:
        clearance = max(float(self.slot_clearance_margin), 1e-3)
        point = np.asarray(point_xy, dtype=float).copy()
        target_pos = np.asarray(target_pos_xy, dtype=float)
        preferred = self._safe_normalize(point - target_pos)
        if np.linalg.norm(preferred) <= 1e-6:
            preferred = np.array([1.0, 0.0], dtype=float)

        if boundary_length is not None:
            point = np.clip(point, clearance, boundary_length - clearance)

        for _ in range(8):
            adjusted = False
            if boundary_length is not None:
                point = np.clip(point, clearance, boundary_length - clearance)
            for obstacle in obstacles:
                delta = point - obstacle.position[:2]
                min_clearance = float(obstacle.radius + clearance)
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

        if boundary_length is not None:
            point = np.clip(point, clearance, boundary_length - clearance)
        return point

    def build_target_slots(self, target, desired_count: int,
                           escape_direction: Optional[np.ndarray] = None,
                           obstacles: Optional[List] = None,
                           boundary_length: Optional[float] = None) -> List[Dict[str, np.ndarray]]:
        if desired_count <= 0:
            return []
        if obstacles is None:
            obstacles = []
        base_dir = self._safe_normalize(
            escape_direction if escape_direction is not None else target.velocity[:2]
        )
        if np.linalg.norm(base_dir) <= 1e-6:
            base_dir = np.array([1.0, 0.0], dtype=float)
        base_angle = float(np.arctan2(base_dir[1], base_dir[0]))
        center = np.asarray(target.position[:2], dtype=float)
        slots = []
        for slot_index, offset_angle in enumerate(self._build_slot_angles(desired_count)):
            angle = base_angle + offset_angle
            raw_position = center + self.slot_radius * np.array(
                [np.cos(angle), np.sin(angle)],
                dtype=float,
            )
            position = self._project_slot_position(
                raw_position,
                center,
                obstacles,
                boundary_length=boundary_length,
            )
            slots.append({
                'slot_index': slot_index,
                'angle': angle,
                'position': position,
            })
        return slots

    def _proximity_score(self, distance: float, scale: float) -> float:
        safe_scale = max(scale, 1e-6)
        return float(np.exp(-((distance / safe_scale) ** 2)))

    def _build_assignment_score(
        self,
        hunter,
        target,
        current_group: List,
        obstacles: List,
        slot_meta: Dict[str, np.ndarray],
        desired_count: int,
        target_weight: float,
        previous_assignments: Optional[Dict[int, Dict[str, object]]],
    ) -> float:
        density_value = self.compute_utility(
            hunter,
            target,
            current_group,
            obstacles,
            w_j=target_weight,
        )
        hunter_pos = np.asarray(hunter.position[:2], dtype=float)
        target_pos = np.asarray(target.position[:2], dtype=float)
        slot_pos = np.asarray(slot_meta['position'], dtype=float)
        target_distance = float(np.linalg.norm(target_pos - hunter_pos))
        slot_distance = float(np.linalg.norm(slot_pos - hunter_pos))

        base_score = (
            self.density_value_weight * density_value
            + self.target_proximity_weight * self._proximity_score(target_distance, self.target_distance_scale)
            + self.slot_proximity_weight * self._proximity_score(slot_distance, self.slot_distance_scale)
        )
        score = target_weight * base_score
        score *= self._coverage_factor(len(current_group), desired_count)

        if previous_assignments:
            prev_assignment = previous_assignments.get(id(hunter))
            if prev_assignment:
                prev_target = prev_assignment.get('target')
                prev_slot_index = prev_assignment.get('slot_index')
                if prev_target is target:
                    score += self.target_inertia_bonus
                    if prev_slot_index == slot_meta['slot_index']:
                        score += self.slot_inertia_bonus
                else:
                    score -= self.switch_penalty

        return float(score)

    def assign_targets(self, hunters: List, targets: List,
                      obstacles: List, target_weights: Optional[Dict[int, float]] = None,
                      previous_assignments: Optional[Dict[int, Dict[str, object]]] = None,
                      target_escape_directions: Optional[Dict[int, np.ndarray]] = None,
                      boundary_length: Optional[float] = None,
                      return_metadata: bool = False) -> Dict[int, object]:
        """
        为所有hunter分配目标，返回 {hunter_id: target} 映射
        
        算法:
        1. 对每个hunter，计算对所有可见target的追击效用
        2. 选择效用最高的target作为追击目标
        
        参数:
            hunters: Hunter对象列表
            targets: Target对象列表
            obstacles: Obstacle对象列表
            
        返回:
            字典 {hunter对象id: target对象}
        """
        assignments = {}
        if not hunters or not targets:
            return assignments

        desired_coverages = self._compute_desired_coverages(len(hunters), len(targets))
        target_groups = {target: [] for target in targets}
        remaining_hunters = list(hunters)
        ordered_targets = sorted(
            targets,
            key=lambda target: self._target_weight(target, target_weights),
            reverse=True,
        )
        target_slots = {
            target: self.build_target_slots(
                target,
                desired_coverages[target_index],
                None if target_escape_directions is None else target_escape_directions.get(id(target)),
                obstacles=obstacles,
                boundary_length=boundary_length,
            )
            for target_index, target in enumerate(ordered_targets)
        }
        metadata = {}

        # First pass: fill each target towards a balanced slot plan.
        for coverage_round in range(max(desired_coverages) if desired_coverages else 0):
            for target_index, target in enumerate(ordered_targets):
                desired_count = desired_coverages[target_index]
                current_group = target_groups[target]
                if len(current_group) >= desired_count or not remaining_hunters:
                    continue
                slot_plan = target_slots[target]
                if coverage_round >= len(slot_plan):
                    continue
                slot_meta = slot_plan[coverage_round]

                best_hunter = None
                best_score = -float('inf')
                target_weight = self._target_weight(target, target_weights)
                for hunter in remaining_hunters:
                    score = self._build_assignment_score(
                        hunter,
                        target,
                        current_group,
                        obstacles,
                        slot_meta,
                        desired_count,
                        target_weight,
                        previous_assignments,
                    )
                    if score > best_score:
                        best_score = score
                        best_hunter = hunter

                if best_hunter is not None:
                    assignments[id(best_hunter)] = target
                    target_groups[target].append(best_hunter)
                    remaining_hunters.remove(best_hunter)
                    metadata[id(best_hunter)] = {
                        'target': target,
                        'slot_index': slot_meta['slot_index'],
                        'slot_position': np.asarray(slot_meta['position'], dtype=float).copy(),
                        'score': float(best_score),
                    }

        # Second pass: assign any leftovers using utility with overload damping.
        for hunter in remaining_hunters:
            best_target = None
            best_slot_meta = None
            best_score = -float('inf')
            for target_index, target in enumerate(ordered_targets):
                current_group = target_groups[target]
                target_weight = self._target_weight(target, target_weights)
                desired_count = desired_coverages[target_index]
                slot_plan = target_slots[target]
                if slot_plan:
                    slot_meta = slot_plan[min(len(current_group), len(slot_plan) - 1)]
                else:
                    slot_meta = {
                        'slot_index': 0,
                        'position': np.asarray(target.position[:2], dtype=float).copy(),
                    }
                score = self._build_assignment_score(
                    hunter,
                    target,
                    current_group,
                    obstacles,
                    slot_meta,
                    desired_count,
                    target_weight,
                    previous_assignments,
                )
                if score > best_score:
                    best_score = score
                    best_target = target
                    best_slot_meta = slot_meta

            if best_target is not None and best_slot_meta is not None:
                assignments[id(hunter)] = best_target
                target_groups[best_target].append(hunter)
                metadata[id(hunter)] = {
                    'target': best_target,
                    'slot_index': best_slot_meta['slot_index'],
                    'slot_position': np.asarray(best_slot_meta['position'], dtype=float).copy(),
                    'score': float(best_score),
                }

        if return_metadata:
            return metadata
        return assignments
