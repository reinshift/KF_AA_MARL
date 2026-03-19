"""
参考速度计算器 (Reference Velocity Calculator)

为target计算参考速度矢量，指导逃逸行为。
参考速度由逃逸向量和避障向量组成，帮助target在hunter之间穿行并避开障碍物。
"""

import numpy as np
from typing import List, Optional


class ReferenceVelocityCalculator:
    """
    参考速度计算器，为target计算逃逸和避障的参考速度
    
    参考速度公式: v_ref = normalize(v1) + normalize(v2)
    其中:
        - v1: 逃逸向量，远离hunter的方向
        - v2: 避障向量，基于激光雷达数据选择最安全的方向
    """
    
    def __init__(self, perception_range: float = 0.5):
        """
        初始化参考速度计算器
        
        参数:
            perception_range: target的感知范围，只考虑此范围内的hunter
        """
        self.perception_range = perception_range
    
    def compute_escape_vector(self, target_pos: np.ndarray,
                             hunter_positions: List[np.ndarray]) -> np.ndarray:
        """
        计算逃逸向量 v1
        
        v1 = -Σ(x_hunter_i - x_target) / ||Σ(x_hunter_i - x_target)||
        
        逃逸向量指向远离所有hunter质心的方向
        
        参数:
            target_pos: 目标位置 [x, y, z]
            hunter_positions: 感知范围内的hunter位置列表
            
        返回:
            逃逸向量 [vx, vy]，如果没有hunter则返回零向量
        """
        if len(hunter_positions) == 0:
            return np.zeros(2)
        
        # 计算所有hunter到target的方向向量之和（2D）
        direction_sum = np.zeros(2)
        for hunter_pos in hunter_positions:
            direction_sum += (hunter_pos[:2] - target_pos[:2])
        
        # 逃逸向量是反方向
        escape_vector = -direction_sum
        
        return escape_vector
    
    def compute_avoidance_vector(self, laser_data: np.ndarray,
                                laser_angles: np.ndarray,
                                max_range: Optional[float] = None) -> np.ndarray:
        """
        计算避障向量 v2，基于激光雷达数据

        将每条激光束的“缩短量”视为来自该方向的占据/危险强度：
            shortened_i = max_range - distance_i
        然后把这些缩短后的射线向量相加，取反得到斥力方向。
        如果所有激光都未缩短，则返回零向量。
        
        参数:
            laser_data: 激光雷达距离数据，形状 (num_lasers,)
            laser_angles: 激光雷达角度数据，形状 (num_lasers,)
            max_range: 激光最大量程，用于计算每条射线的缩短量
            
        返回:
            避障向量 [vx, vy]
        """
        if len(laser_data) == 0:
            return np.zeros(2)

        laser_data = np.asarray(laser_data, dtype=float)
        laser_angles = np.asarray(laser_angles, dtype=float)
        if max_range is None:
            max_range = float(np.max(laser_data)) if len(laser_data) > 0 else 0.0
        max_range = max(float(max_range), 1e-6)

        shortened = np.clip(max_range - laser_data, 0.0, max_range)
        if np.all(shortened <= 1e-9):
            return np.zeros(2)

        beam_dirs = np.column_stack((np.cos(laser_angles), np.sin(laser_angles)))
        occupancy_vector = np.sum(shortened[:, None] * beam_dirs, axis=0)
        repulsion_vector = -occupancy_vector
        repulsion_norm = np.linalg.norm(repulsion_vector)
        if repulsion_norm <= 1e-9:
            return np.zeros(2)

        return repulsion_vector / repulsion_norm

    def compute_reference_velocity(self, target, hunters: List,
                                    escape_zone_center: np.ndarray = None) -> np.ndarray:
        """
        计算参考速度 v_ref = normalize(v1) + normalize(v2) + w3 * normalize(v3)

        v1: 逃逸向量（远离hunter）
        v2: 避障向量（雷达最安全方向）
        v3: 出口吸引力（朝逃逸区域）

        参数:
            target: Target对象
            hunters: Hunter对象列表
            escape_zone_center: 逃逸区域中心坐标 [x,y]，如果提供则加入出口吸引力

        返回:
            参考速度向量 [vx, vy]
        """
        # 找到感知范围内的hunters
        hunters_in_range = []
        for hunter in hunters:
            distance = np.linalg.norm(hunter.position[:2] - target.position[:2])
            if distance < self.perception_range:
                hunters_in_range.append(hunter)

        # 计算逃逸向量
        if len(hunters_in_range) > 0:
            hunter_positions = [h.position for h in hunters_in_range]
            v1 = self.compute_escape_vector(target.position, hunter_positions)
        else:
            # 没有hunter在感知范围内，使用随机方向
            angle = np.random.uniform(0, 2 * np.pi)
            v1 = np.array([np.cos(angle), np.sin(angle)])

        # 计算避障向量
        v2 = self.compute_avoidance_vector(
            target.lasers, target.lidar.angles, max_range=target.lidar.max_detect_d
        )

        # 计算出口吸引力（近距离大，远距离小）
        v3 = np.zeros(2)
        w3 = 0.0
        if escape_zone_center is not None:
            escape_dir = escape_zone_center - target.position[:2]
            escape_dist = np.linalg.norm(escape_dir)
            if escape_dist > 1e-6:
                v3 = escape_dir / escape_dist
                # 对角线长度作为归一化参考
                diag = np.sqrt(2) * 2.0  # 地图对角线约2.83
                # 远距离权重低，近距离权重高
                w3 = 0.5 * max(0.0, 1.0 - escape_dist / max(diag, 1e-6))

        # 合成各分力（避障权重提高到1.5）
        epsilon = 1e-6
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        v_ref = np.zeros(2)
        if v1_norm > epsilon:
            v_ref += v1 / v1_norm
        if v2_norm > epsilon:
            v_ref += 1.5 * (v2 / v2_norm)
        if w3 > 0:
            v_ref += w3 * v3

        # 如果合力为零，保持当前速度方向
        v_ref_norm = np.linalg.norm(v_ref)
        if v_ref_norm < epsilon:
            current_vel_norm = np.linalg.norm(target.velocity[:2])
            if current_vel_norm > epsilon:
                return target.velocity[:2] / current_vel_norm
            else:
                return np.array([1.0, 0.0])

        return v_ref
    
    def is_inside_obstacle(self, position: np.ndarray,
                          obstacles: List) -> bool:
        """
        检测位置是否在障碍物内部
        
        如果位置到某个障碍物中心的距离小于障碍物半径，则认为在障碍物内部
        
        参数:
            position: 位置 [x, y, z]
            obstacles: Obstacle对象列表
            
        返回:
            True如果在障碍物内部，否则False
        """
        for obstacle in obstacles:
            # 计算到障碍物中心的距离（2D）
            distance = np.linalg.norm(position[:2] - obstacle.position[:2])
            
            # 检查是否在障碍物半径内
            if distance < obstacle.radius:
                return True
        
        return False
