"""
角色分配器实现
为追击同一target的hunter分配chaser或interceptor角色

使用卡尔曼滤波器预测target未来位置，根据hunter与target的相对位置、
距离和速度动态分配角色。
"""

import numpy as np
from typing import Callable, Dict, List, Optional
from kalman_filter import KalmanFilter


class RoleAssigner:
    """
    角色分配器，为追击同一target的hunter分配角色
    
    角色类型:
    - chaser: 追踪者，直接追踪target的当前位置
    - interceptor: 拦截者，预测并拦截target未来位置
    """
    
    def __init__(self, dt: float = 0.5, process_noise: float = 0.01,
                 measurement_noise: float = 0.1,
                 min_group_size_for_interceptor: int = 3,
                 max_interceptors_per_target: int = 1,
                 min_target_speed: float = 0.025,
                 min_interceptor_distance: float = 0.12,
                 max_interceptor_distance: float = 0.30,
                 prediction_steps: int = 4,
                 interceptor_persistence_bonus: float = 0.12):
        """
        初始化角色分配器
        
        参数:
            dt: 时间步长
            process_noise: 卡尔曼滤波器的过程噪声协方差
            measurement_noise: 卡尔曼滤波器的测量噪声协方差
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.min_group_size_for_interceptor = min_group_size_for_interceptor
        self.max_interceptors_per_target = max_interceptors_per_target
        self.min_target_speed = min_target_speed
        self.min_interceptor_distance = min_interceptor_distance
        self.max_interceptor_distance = max_interceptor_distance
        self.prediction_steps = prediction_steps
        self.interceptor_persistence_bonus = interceptor_persistence_bonus
        
        # 为每个target维护一个卡尔曼滤波器
        # key: target的id (使用id(target))
        # value: KalmanFilter实例
        self.kalman_filters: Dict[int, KalmanFilter] = {}

    @staticmethod
    def _safe_normalize(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= eps:
            return np.zeros(2, dtype=float)
        return np.asarray(vector, dtype=float) / norm

    @staticmethod
    def _perpendicular(vector: np.ndarray) -> np.ndarray:
        return np.array([-vector[1], vector[0]], dtype=float)
    
    def assign_roles(self, hunters: List, target) -> Dict[int, str]:
        """
        为追击同一target的hunters分配角色
        
        策略:
        1. 计算每个hunter到target的距离和相对角度
        2. 预测target未来位置
        3. 如果hunter在target前方且距离适中 -> interceptor
        4. 否则 -> chaser
        
        参数:
            hunters: 追击同一target的hunter列表
            target: 被追击的target
        
        返回:
            {hunter_id: 'chaser' or 'interceptor'} 映射
        """
        if not hunters:
            return {}

        roles = {id(hunter): 'chaser' for hunter in hunters}
        target_velocity = target.velocity[:2]
        target_speed = np.linalg.norm(target_velocity)
        if (
            len(hunters) < self.min_group_size_for_interceptor
            or target_speed < self.min_target_speed
            or self.max_interceptors_per_target <= 0
        ):
            return roles

        predicted_pos = self.predict_target_position(target, steps_ahead=self.prediction_steps)
        target_pos = np.asarray(target.position[:2], dtype=float)
        travel_axis = predicted_pos - target_pos
        if np.linalg.norm(travel_axis) <= 1e-6:
            travel_axis = target_velocity
        target_dir = self._safe_normalize(travel_axis)
        if np.linalg.norm(target_dir) <= 1e-6:
            return roles
        lateral_dir = self._perpendicular(target_dir)
        target_lead_distance = max(
            float(np.linalg.norm(predicted_pos - target_pos)),
            self.min_interceptor_distance,
        )
        interceptor_candidates = []
        fallback_candidates = []

        for hunter in hunters:
            hunter_pos = np.asarray(hunter.position[:2], dtype=float)
            rel_to_target = hunter_pos - target_pos
            distance = np.linalg.norm(rel_to_target)
            if distance <= 1e-6:
                continue

            predicted_distance = np.linalg.norm(predicted_pos - hunter.position[:2])
            forward_progress = float(np.dot(rel_to_target, target_dir))
            lateral_offset = abs(float(np.dot(rel_to_target, lateral_dir)))
            arrival_margin = target_lead_distance - predicted_distance
            bracket_score = -abs(
                distance - 0.5 * (self.min_interceptor_distance + self.max_interceptor_distance)
            )
            persistence_bonus = (
                self.interceptor_persistence_bonus
                if getattr(hunter, 'role', None) == 'interceptor'
                else 0.0
            )
            candidate_score = (
                1.35 * arrival_margin
                + 0.85 * forward_progress
                - 0.45 * lateral_offset
                + 0.30 * bracket_score
                + persistence_bonus
            )
            fallback_candidates.append((candidate_score, hunter))
            if distance < self.min_interceptor_distance * 0.75:
                continue
            if (
                distance > self.max_interceptor_distance * 1.5
                and predicted_distance > self.max_interceptor_distance * 1.5
            ):
                continue
            interceptor_candidates.append((candidate_score, hunter))

        candidate_pool = interceptor_candidates if interceptor_candidates else fallback_candidates
        max_interceptors = min(
            self.max_interceptors_per_target,
            max(0, len(hunters) - 2),
            len(candidate_pool),
        )
        candidate_pool.sort(key=lambda item: item[0], reverse=True)
        for _, hunter in candidate_pool[:max_interceptors]:
            roles[id(hunter)] = 'interceptor'

        return roles
    
    def predict_target_position(self, target, steps_ahead: int = 5) -> np.ndarray:
        """
        使用卡尔曼滤波预测target未来位置
        
        参数:
            target: 要预测的target
            steps_ahead: 预测的步数
        
        返回:
            预测的未来位置 [px, py]
        """
        target_id = id(target)
        
        # 如果该target还没有卡尔曼滤波器，创建一个
        if target_id not in self.kalman_filters:
            kf = KalmanFilter(
                dt=self.dt,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise
            )
            # 初始化滤波器
            kf.initialize(
                position=target.position[:2],
                velocity=target.velocity[:2]
            )
            self.kalman_filters[target_id] = kf
        else:
            kf = self.kalman_filters[target_id]

        predicted_pos = kf.predict_future(steps_ahead=steps_ahead)
        if np.linalg.norm(kf.get_velocity()) <= 1e-6:
            predicted_pos = target.position[:2] + target.velocity[:2] * self.dt * steps_ahead

        return predicted_pos
    
    def update_kalman_filter(
        self,
        target_id: int,
        measurement: np.ndarray,
        velocity: Optional[np.ndarray] = None,
    ):
        """
        更新指定target的卡尔曼滤波器
        
        参数:
            target_id: target的id (使用id(target))
            measurement: 观测值 [px, py]
        """
        if velocity is not None:
            velocity = np.asarray(velocity, dtype=np.float64).flatten()
            if velocity.shape[0] != 2:
                raise ValueError(f"Velocity must have 2 elements, got {velocity.shape[0]}")

        if target_id not in self.kalman_filters:
            # 如果还没有该target的滤波器，创建一个
            kf = KalmanFilter(
                dt=self.dt,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise
            )
            kf.initialize(position=measurement, velocity=velocity)
            self.kalman_filters[target_id] = kf
        else:
            kf = self.kalman_filters[target_id]
            kf.predict()
            kf.update(measurement)
            if velocity is not None:
                kf.x[2:] = 0.6 * kf.x[2:] + 0.4 * velocity
    
    def get_target_position_for_hunter(
        self,
        hunter,
        target,
        role: str,
        project_fn: Optional[Callable[[np.ndarray, object], np.ndarray]] = None,
    ) -> np.ndarray:
        """
        根据hunter的角色获取目标位置
        
        参数:
            hunter: hunter对象
            target: target对象
            role: hunter的角色 ('chaser' 或 'interceptor')
        
        返回:
            目标位置 [px, py, pz]，其中z坐标与target当前高度相同
        """
        if role == 'chaser':
            # chaser追踪target当前位置
            return target.position.copy()
        elif role == 'interceptor':
            # interceptor追踪预测的未来位置
            predicted_pos_2d = self.predict_target_position(target, steps_ahead=self.prediction_steps)
            if project_fn is not None:
                predicted_pos_2d = np.asarray(project_fn(predicted_pos_2d, target), dtype=float)
            # 添加z坐标（保持与target当前高度相同）
            target_pos_3d = np.array([
                predicted_pos_2d[0],
                predicted_pos_2d[1],
                target.position[2]
            ])
            return target_pos_3d
        else:
            raise ValueError(f"Unknown role: {role}")
    
    def reset(self):
        """
        重置所有卡尔曼滤波器
        通常在环境reset时调用
        """
        self.kalman_filters.clear()
