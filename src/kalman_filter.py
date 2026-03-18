"""
卡尔曼滤波器实现
用于预测target的未来位置

状态向量: x = [px, py, vx, vy]^T
观测向量: z = [px, py]^T
"""

import numpy as np


class KalmanFilter:
    """
    卡尔曼滤波器，用于预测target的位置和速度
    
    状态向量包含位置和速度: [px, py, vx, vy]
    观测向量只包含位置: [px, py]
    """
    
    def __init__(self, dt: float = 0.1, process_noise: float = 0.01, measurement_noise: float = 0.1):
        """
        初始化卡尔曼滤波器
        
        参数:
            dt: 时间步长
            process_noise: 过程噪声协方差
            measurement_noise: 测量噪声协方差
        """
        self.dt = dt
        
        # 状态转移矩阵 F
        # x_{k+1} = F * x_k + w_k
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        # 观测矩阵 H
        # z_k = H * x_k + v_k
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)
        
        # 过程噪声协方差矩阵 Q
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        
        # 测量噪声协方差矩阵 R
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        
        # 状态估计向量
        self.x = np.zeros(4, dtype=np.float64)
        
        # 估计协方差矩阵 P
        self.P = np.eye(4, dtype=np.float64)
        
        # 用于协方差重置的计数器
        self.update_count = 0
        self.reset_interval = 100  # 每100次更新重置一次协方差
    
    def predict(self) -> np.ndarray:
        """
        预测步骤
        
        返回:
            预测的位置 [px, py]
        """
        # 状态预测: x_k|k-1 = F * x_k-1|k-1
        self.x = self.F @ self.x
        
        # 协方差预测: P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # 返回预测的位置
        return self.x[:2].copy()
    
    def update(self, measurement: np.ndarray):
        """
        更新步骤（使用Joseph形式提高数值稳定性）
        
        参数:
            measurement: 观测值 [px, py]
        """
        # 确保measurement是正确的形状
        measurement = np.asarray(measurement, dtype=np.float64).flatten()
        if measurement.shape[0] != 2:
            raise ValueError(f"Measurement must have 2 elements, got {measurement.shape[0]}")
        
        # 创新（innovation）: y = z - H * x
        y = measurement - self.H @ self.x
        
        # 创新协方差: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # 卡尔曼增益: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 状态更新: x = x + K * y
        self.x = self.x + K @ y
        
        # 使用Joseph形式更新协方差，提高数值稳定性
        # P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        I_KH = np.eye(4, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # 防止P过小，添加小的正定项
        self.P = self.P + np.eye(4, dtype=np.float64) * 1e-8
        
        # 协方差重置机制，防止累积误差
        self.update_count += 1
        if self.update_count >= self.reset_interval:
            self._reset_covariance()
            self.update_count = 0
    
    def _reset_covariance(self):
        """
        重置协方差矩阵，防止累积误差
        保持当前状态估计，但重置不确定性
        """
        # 重置为初始不确定性
        self.P = np.eye(4, dtype=np.float64)
    
    def predict_future(self, steps_ahead: int = 1) -> np.ndarray:
        """
        预测未来多步的位置
        
        参数:
            steps_ahead: 预测的步数
        
        返回:
            预测的未来位置 [px, py]
        """
        if steps_ahead < 1:
            raise ValueError("steps_ahead must be at least 1")
        
        # 保存当前状态
        x_current = self.x.copy()
        P_current = self.P.copy()
        
        # 多步预测
        x_pred = x_current.copy()
        for _ in range(steps_ahead):
            x_pred = self.F @ x_pred
        
        # 恢复当前状态（不改变滤波器状态）
        # 注意：这里不需要恢复，因为我们没有修改self.x和self.P
        
        return x_pred[:2].copy()
    
    def initialize(self, position: np.ndarray, velocity: np.ndarray = None):
        """
        初始化滤波器状态
        
        参数:
            position: 初始位置 [px, py]
            velocity: 初始速度 [vx, vy]，如果为None则设为0
        """
        position = np.asarray(position, dtype=np.float64).flatten()
        if position.shape[0] != 2:
            raise ValueError(f"Position must have 2 elements, got {position.shape[0]}")
        
        if velocity is None:
            velocity = np.zeros(2, dtype=np.float64)
        else:
            velocity = np.asarray(velocity, dtype=np.float64).flatten()
            if velocity.shape[0] != 2:
                raise ValueError(f"Velocity must have 2 elements, got {velocity.shape[0]}")
        
        self.x = np.array([position[0], position[1], velocity[0], velocity[1]], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)
        self.update_count = 0
    
    def get_state(self) -> np.ndarray:
        """
        获取当前状态估计
        
        返回:
            状态向量 [px, py, vx, vy]
        """
        return self.x.copy()
    
    def get_position(self) -> np.ndarray:
        """
        获取当前位置估计
        
        返回:
            位置向量 [px, py]
        """
        return self.x[:2].copy()
    
    def get_velocity(self) -> np.ndarray:
        """
        获取当前速度估计
        
        返回:
            速度向量 [vx, vy]
        """
        return self.x[2:].copy()
