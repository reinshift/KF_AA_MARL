"""
Reference velocity calculation for target agents.
"""

from typing import List, Optional

import numpy as np


class ReferenceVelocityCalculator:
    """
    Compute a locally feasible target guidance velocity from:
    - hunter repulsion
    - obstacle repulsion
    - exit attraction
    """

    def __init__(self, perception_range: float = 0.5):
        self.perception_range = perception_range

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _safe_normalize(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= eps:
            return np.zeros(2, dtype=float)
        return np.asarray(vector, dtype=float) / norm

    def _compute_feasible_exit_guidance(
        self,
        laser_data: np.ndarray,
        laser_angles: np.ndarray,
        exit_direction: np.ndarray,
        max_range: float,
    ) -> tuple[np.ndarray, float]:
        """
        Project exit attraction onto a locally feasible ray and attenuate its weight
        when the direct path is obstructed.
        """
        exit_norm = np.linalg.norm(exit_direction)
        if exit_norm <= 1e-6 or len(laser_data) == 0:
            return np.zeros(2), 0.0

        laser_data = np.asarray(laser_data, dtype=float)
        laser_angles = np.asarray(laser_angles, dtype=float)
        max_range = max(float(max_range), 1e-6)
        exit_unit = exit_direction / exit_norm
        exit_angle = float(np.arctan2(exit_unit[1], exit_unit[0]))

        angle_diffs = np.array([self._wrap_to_pi(angle - exit_angle) for angle in laser_angles], dtype=float)
        direct_idx = int(np.argmin(np.abs(angle_diffs)))
        direct_clearance = float(np.clip(laser_data[direct_idx] / max_range, 0.0, 1.0))

        feasible_threshold = 0.65 * max_range
        feasible_mask = laser_data >= feasible_threshold
        if np.any(feasible_mask):
            feasible_indices = np.where(feasible_mask)[0]
            best_idx = int(feasible_indices[np.argmin(np.abs(angle_diffs[feasible_indices]))])
            projected_angle = float(laser_angles[best_idx])
            projected_clearance = float(np.clip(laser_data[best_idx] / max_range, 0.0, 1.0))
            angular_alignment = max(0.0, np.cos(self._wrap_to_pi(projected_angle - exit_angle)))
            # Nearby clear gaps should still produce a meaningful exit pull even when
            # the direct ray is blocked, otherwise the target learns to stall in corners.
            feasibility_gate = (
                0.20 * direct_clearance
                + 0.55 * projected_clearance
                + 0.25 * angular_alignment
            )
        else:
            best_idx = direct_idx
            projected_angle = float(laser_angles[best_idx])
            feasibility_gate = 0.35 * (direct_clearance ** 2)

        projected_vector = np.array([np.cos(projected_angle), np.sin(projected_angle)], dtype=float)
        return projected_vector, float(np.clip(feasibility_gate, 0.0, 1.0))

    def compute_escape_vector(self, target_pos: np.ndarray, hunter_positions: List[np.ndarray]) -> np.ndarray:
        if len(hunter_positions) == 0:
            return np.zeros(2)

        direction_sum = np.zeros(2, dtype=float)
        for hunter_pos in hunter_positions:
            delta = target_pos[:2] - hunter_pos[:2]
            dist = float(np.linalg.norm(delta))
            if dist <= 1e-6:
                continue
            # Boids-style separation: nearby hunters exert a much stronger repulsion.
            direction_sum += delta / max(dist * dist, 1e-6)

        return direction_sum

    def compute_avoidance_vector(
        self,
        laser_data: np.ndarray,
        laser_angles: np.ndarray,
        max_range: Optional[float] = None,
    ) -> np.ndarray:
        """
        Sum shortened laser beams as occupancy vectors and negate them to obtain
        a repulsion direction. If no beams are shortened, return zero.
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

    def _project_direction_to_feasible_ray(
        self,
        laser_data: np.ndarray,
        laser_angles: np.ndarray,
        desired_direction: np.ndarray,
        max_range: float,
    ) -> tuple[np.ndarray, float]:
        """
        Keep a steering direction smooth in open space, but project it toward a
        nearby clear ray when the direct direction is blocked.
        """
        desired_unit = self._safe_normalize(desired_direction)
        if np.linalg.norm(desired_unit) <= 1e-9 or len(laser_data) == 0:
            return desired_unit, 0.0

        laser_data = np.asarray(laser_data, dtype=float)
        laser_angles = np.asarray(laser_angles, dtype=float)
        max_range = max(float(max_range), 1e-6)
        desired_angle = float(np.arctan2(desired_unit[1], desired_unit[0]))
        angle_diffs = np.array(
            [self._wrap_to_pi(angle - desired_angle) for angle in laser_angles],
            dtype=float,
        )

        direct_idx = int(np.argmin(np.abs(angle_diffs)))
        direct_clearance = float(np.clip(laser_data[direct_idx] / max_range, 0.0, 1.0))
        if direct_clearance >= 0.8:
            return desired_unit, 1.0

        feasible_mask = laser_data >= 0.6 * max_range
        if np.any(feasible_mask):
            feasible_indices = np.where(feasible_mask)[0]
            best_idx = int(feasible_indices[np.argmin(np.abs(angle_diffs[feasible_indices]))])
        else:
            best_idx = int(np.argmax(laser_data))

        projected_angle = float(laser_angles[best_idx])
        projected_direction = np.array(
            [np.cos(projected_angle), np.sin(projected_angle)],
            dtype=float,
        )
        angular_alignment = max(
            0.0,
            np.cos(self._wrap_to_pi(projected_angle - desired_angle)),
        )
        projection_mix = float(np.clip(1.0 - direct_clearance, 0.0, 1.0))
        corrected = self._safe_normalize(
            (1.0 - projection_mix) * desired_unit + projection_mix * projected_direction
        )
        feasibility_gate = float(
            np.clip(
                direct_clearance * 0.4 + (1.0 - direct_clearance) * (0.3 + 0.7 * angular_alignment),
                0.0,
                1.0,
            )
        )
        return corrected, feasibility_gate

    def compute_reference_velocity(
        self,
        target,
        hunters: List,
        escape_zone_center: np.ndarray = None,
        hunter_weight_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Boids-style steering with four components:
        - separation from nearby hunters
        - obstacle repulsion
        - feasible exit attraction
        - inertia from current velocity
        """
        hunters_in_range = []
        nearest_hunter_dist = self.perception_range
        if hunter_weight_scale > 1e-6:
            for hunter in hunters:
                distance = np.linalg.norm(hunter.position[:2] - target.position[:2])
                if distance < self.perception_range:
                    hunters_in_range.append(hunter)
                    nearest_hunter_dist = min(nearest_hunter_dist, float(distance))

        if len(hunters_in_range) > 0:
            hunter_positions = [hunter.position for hunter in hunters_in_range]
            v1 = self.compute_escape_vector(target.position, hunter_positions)
        else:
            v1 = np.zeros(2)

        v2 = self.compute_avoidance_vector(
            target.lasers,
            target.lidar.angles,
            max_range=target.lidar.max_detect_d,
        )

        v3 = np.zeros(2)
        w3 = 0.0
        if escape_zone_center is not None:
            escape_dir = escape_zone_center - target.position[:2]
            escape_dist = np.linalg.norm(escape_dir)
            if escape_dist > 1e-6:
                v3, exit_gate = self._compute_feasible_exit_guidance(
                    target.lasers,
                    target.lidar.angles,
                    escape_dir,
                    target.lidar.max_detect_d,
                )
                diag = np.sqrt(2.0) * 2.0
                progress_pressure = max(0.0, 1.0 - escape_dist / max(diag, 1e-6))
                w3 = exit_gate * (0.55 + 0.55 * progress_pressure)

        current_velocity = np.asarray(target.velocity[:2], dtype=float)
        v4 = self._safe_normalize(current_velocity)

        max_range = max(float(target.lidar.max_detect_d), 1e-6)
        min_laser = float(np.min(target.lasers)) if len(target.lasers) > 0 else max_range
        free_space_ratio = float(np.clip(np.mean(target.lasers) / max_range, 0.0, 1.0))
        obstacle_pressure = float(np.clip(1.0 - min_laser / max_range, 0.0, 1.0))
        hunter_pressure = 0.0
        if len(hunters_in_range) > 0 and self.perception_range > 1e-6:
            hunter_pressure = float(
                np.clip(1.0 - nearest_hunter_dist / self.perception_range, 0.0, 1.0)
            )

        w_sep = hunter_weight_scale * (0.5 + 2.0 * hunter_pressure)
        w_avoid = 0.85 + 1.85 * obstacle_pressure
        w_goal = w3 * (0.65 + 0.55 * (1.0 - hunter_pressure)) * (0.35 + 0.65 * free_space_ratio)
        w_inertia = 0.25 + 0.45 * (1.0 - obstacle_pressure)

        desired = np.zeros(2, dtype=float)
        desired += w_sep * self._safe_normalize(v1)
        desired += w_avoid * self._safe_normalize(v2)
        desired += w_goal * self._safe_normalize(v3)
        desired += w_inertia * v4

        desired_dir = self._safe_normalize(desired)
        if np.linalg.norm(desired_dir) <= 1e-9:
            if np.linalg.norm(v4) > 1e-9:
                return v4
            if np.linalg.norm(v3) > 1e-9:
                return self._safe_normalize(v3)
            return np.array([1.0, 0.0], dtype=float)

        feasible_dir, feasibility_gate = self._project_direction_to_feasible_ray(
            target.lasers,
            target.lidar.angles,
            desired_dir,
            max_range=max_range,
        )

        if np.linalg.norm(feasible_dir) <= 1e-9:
            return desired_dir

        if np.linalg.norm(v4) > 1e-9:
            inertia_blend = 0.2 + 0.3 * feasibility_gate * (1.0 - obstacle_pressure)
            feasible_dir = self._safe_normalize(
                (1.0 - inertia_blend) * feasible_dir + inertia_blend * v4
            )

        return feasible_dir

    def is_inside_obstacle(self, position: np.ndarray, obstacles: List) -> bool:
        for obstacle in obstacles:
            distance = np.linalg.norm(position[:2] - obstacle.position[:2])
            if distance < obstacle.radius:
                return True
        return False
