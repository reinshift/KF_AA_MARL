import math
import numpy as np

def vector_length(v):
    return math.hypot(v[0], v[1])

def unit_vector(v):
    length = vector_length(v)
    if length == 0:
        return (0, 0)
    return (v[0]/length, v[1]/length)

def cross_product(v1, v2):
    return v1[0]*v2[1] - v1[1]*v2[0]

def ray_segment_intersection(ray_origin, ray_dir, seg_start, seg_end):
    seg_dir = (seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
    a1 = -seg_dir[1]
    b1 = seg_dir[0]
    c1 = a1*seg_start[0] + b1*seg_start[1]
    a2 = -ray_dir[1]
    b2 = ray_dir[0]
    c2 = a2*ray_origin[0] + b2*ray_origin[1]
    det = a1*b2 - a2*b1
    if det == 0:
        return None
    x = (b2*c1 - b1*c2) / det
    y = (a1*c2 - a2*c1) / det
    if (min(seg_start[0], seg_end[0]) - 1e-8 <= x <= max(seg_start[0], seg_end[0]) + 1e-8) and \
       (min(seg_start[1], seg_end[1]) - 1e-8 <= y <= max(seg_start[1], seg_end[1]) + 1e-8):
        t = (x - ray_origin[0])/ray_dir[0] if ray_dir[0] != 0 else (y - ray_origin[1])/ray_dir[1]
        if t >= 0:
            return (x, y)
    return None

def compute_histogram(target_pos, multi_hunter_pos, distance, max_range=0.2,
                      boundary_length=None, obstacles=None):
    '''
    Just like VFH (Vector Field Histogram)
    Now also considers boundary walls and obstacles as blocking.
    boundary_length: if provided, treat [0, boundary_length] square as walls
    obstacles: list of obstacle objects with _return_obs_info() -> (x,y,z,r,h)
    '''
    tx, ty = target_pos
    angles = np.arange(0, 2 * math.pi, math.pi / 16)  # 32角度
    histogram = []

    # 边界宽容系数：稍微缩小边界的有效阻挡范围，给target极限逃生空间
    boundary_margin = 0.03  # target贴边时仍有一小段"缝隙"

    for angle in angles:
        ray_dir = (math.cos(angle), math.sin(angle))
        min_distance = max_range

        # 检测hunter阻挡
        for hunter in multi_hunter_pos:
            hx, hy = hunter
            dx = tx - hx
            dy = ty - hy
            perp_v1 = (-dy, dx)
            perp_unit1 = unit_vector(perp_v1)
            seg_start = (hx + perp_unit1[0] * distance / 2, hy + perp_unit1[1] * distance / 2)
            seg_end = (hx - perp_unit1[0] * distance / 2, hy - perp_unit1[1] * distance / 2)
            intersection = ray_segment_intersection(target_pos, ray_dir, seg_start, seg_end)
            if intersection is not None:
                dist = math.hypot(intersection[0] - tx, intersection[1] - ty)
                if dist < min_distance:
                    min_distance = dist

        # 检测边界阻挡（带宽容）
        if boundary_length is not None:
            effective_min = boundary_margin
            effective_max = boundary_length - boundary_margin
            # 四条边界线段
            boundary_segments = [
                ((effective_min, effective_min), (effective_max, effective_min)),  # 下边
                ((effective_max, effective_min), (effective_max, effective_max)),  # 右边
                ((effective_max, effective_max), (effective_min, effective_max)),  # 上边
                ((effective_min, effective_max), (effective_min, effective_min)),  # 左边
            ]
            for seg_s, seg_e in boundary_segments:
                intersection = ray_segment_intersection(target_pos, ray_dir, seg_s, seg_e)
                if intersection is not None:
                    dist = math.hypot(intersection[0] - tx, intersection[1] - ty)
                    if dist < min_distance:
                        min_distance = dist

        # 检测障碍物阻挡（圆柱体近似为正多边形线段）
        if obstacles is not None:
            for obs in obstacles:
                cx, cy, cz, r, h = obs._return_obs_info()
                # 用8段线段近似圆
                num_segs = 8
                for k in range(num_segs):
                    a1 = 2 * math.pi * k / num_segs
                    a2 = 2 * math.pi * (k + 1) / num_segs
                    seg_s = (cx + r * math.cos(a1), cy + r * math.sin(a1))
                    seg_e = (cx + r * math.cos(a2), cy + r * math.sin(a2))
                    intersection = ray_segment_intersection(target_pos, ray_dir, seg_s, seg_e)
                    if intersection is not None:
                        dist = math.hypot(intersection[0] - tx, intersection[1] - ty)
                        if dist < min_distance:
                            min_distance = dist

        histogram.append(min_distance if min_distance < max_range else max_range)

    return histogram

def find_largest_clear_band(histogram, angles, max_range=20):
    num_bins = len(histogram)
    angle_step = angles[1] - angles[0]
    
    clear_indices = [i for i in range(num_bins) if abs(histogram[i] - max_range) < 1e-6]
    
    if not clear_indices:
        return None
    
    clear_intervals = []
    start = clear_indices[0]
    for i in range(1, len(clear_indices)):
        if clear_indices[i] != clear_indices[i-1] + 1:
            end = clear_indices[i-1]
            clear_intervals.append((angles[start], angles[end]))
            start = clear_indices[i]
    clear_intervals.append((angles[start], angles[clear_indices[-1]]))
    
    if angles[clear_indices[0]] == 0 and angles[clear_indices[-1]] == 2 * math.pi - angle_step:
        _, first_end = clear_intervals[0]
        last_start, _ = clear_intervals[-1]
        clear_intervals = clear_intervals[1:-1]
        merged_end = first_end + 2 * math.pi
        clear_intervals.append((last_start, merged_end))
    
    largest_interval = None
    largest_size = -1
    for interval in clear_intervals:
        start_angle, end_angle = interval

        if end_angle >= start_angle:
            size = end_angle - start_angle
        else:
            size = (2 * math.pi - start_angle) + end_angle
        
        if size > largest_size:
            largest_size = size
            largest_interval = (start_angle*180/math.pi, end_angle*180/math.pi)
    
    return largest_interval

def isRounded(target_pos, multi_hunter_pos, sense_radius, success_threshold,
              max_range=0.2, boundary_length=None, obstacles=None):
    '''
    Check if the target is rounded by hunters.
    Now considers boundary walls and obstacles as blocking.
    '''
    histogram = compute_histogram(target_pos, multi_hunter_pos, sense_radius * 2, max_range,
                                  boundary_length=boundary_length, obstacles=obstacles)
    angles = np.arange(0, 2 * math.pi, math.pi / 16)
    largest_escape_interval = find_largest_clear_band(histogram, angles, max_range)
    if largest_escape_interval is None:
        return True
    else:
        start_angle, end_angle = largest_escape_interval
        interval_size = end_angle - start_angle
        if interval_size < success_threshold:
            return True
        else:
            return False