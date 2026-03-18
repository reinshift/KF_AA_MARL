"""
角色分配器实现
为追击同一target的hunter分配chaser或interceptor角色

使用卡尔曼滤波器预测target未来位置，根据hunter与target的相对位置、
距离和速度动态分配角色。
"""

import numpy as np
from typing import Dict, List, Optional
from kalman_filter import KalmanFilter


class RoleAssigner:
    """
    角色分配器，为追击同一target的hunter分配角色
    
    角色类型:
    - chaser: 追踪者，直接追踪target的当前位置
    - interceptor: 拦截者，预测并拦截target未来位置
    """
    
    def __init__(self, dt: float = 0.5, process_noise: float = 0.01, measurement_noise: float = 0.1):
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
        
        # 为每个target维护一个卡尔曼滤波器
        # key: target的id (使用id(target))
        # value: KalmanFilter实例
        self.kalman_filters: Dict[int, KalmanFilter] = {}
    
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
        
        roles = {}
        
        # 预测target未来位置
        predicted_pos = self.predict_target_position(target, steps_ahead=5)
        
        for hunter in hunters:
            # 计算相对位置
            to_target = target.position[:2] - hunter.position[:2]
            distance = np.linalg.norm(to_target)
            
            # 计算hunter是否在target前方
            # 前方定义：hunter在target运动方向的前方
            target_velocity = target.velocity[:2]
            is_ahead = False
            
            if np.linalg.norm(target_velocity) > 0.01:
                # target有明显速度，判断hunter是否在前方
                target_dir = target_velocity / np.linalg.norm(target_velocity)
                # 如果to_target与target_dir的点积为负，说明hunter在target前方
                dot_product = np.dot(to_target, target_dir)
                is_ahead = dot_product < 0
            
            # 分配角色
            # interceptor条件：在target前方且距离适中（0.1到0.4之间）
            if is_ahead and 0.1 < distance < 0.4:
                roles[id(hunter)] = 'interceptor'
            else:
                roles[id(hunter)] = 'chaser'
        
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
        
        # 获取该target的卡尔曼滤波器
        kf = self.kalman_filters[target_id]
        
        # 预测未来位置
        predicted_pos = kf.predict_future(steps_ahead=steps_ahead)
        
        return predicted_pos
    
    def update_kalman_filter(self, target_id: int, measurement: np.ndarray):
        """
        更新指定target的卡尔曼滤波器
        
        参数:
            target_id: target的id (使用id(target))
            measurement: 观测值 [px, py]
        """
        if target_id not in self.kalman_filters:
            # 如果还没有该target的滤波器，创建一个
            kf = KalmanFilter(
                dt=self.dt,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise
            )
            # 使用测量值初始化（速度设为0）
            kf.initialize(position=measurement, velocity=None)
            self.kalman_filters[target_id] = kf
        else:
            # 更新现有滤波器
            kf = self.kalman_filters[target_id]
            kf.update(measurement)
    
    def get_target_position_for_hunter(self, hunter, target, role: str) -> np.ndarray:
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
            predicted_pos_2d = self.predict_target_position(target, steps_ahead=5)
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
