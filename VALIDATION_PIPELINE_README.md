# 验证流水线使用指南 (Validation Pipeline User Guide)

## 目录 (Table of Contents)

- [简介 (Introduction)](#简介-introduction)
- [快速开始 (Quick Start)](#快速开始-quick-start)
- [配置文件 (Configuration File)](#配置文件-configuration-file)
- [运行验证 (Running Validation)](#运行验证-running-validation)
- [输出结果 (Output Results)](#输出结果-output-results)
- [高级用法 (Advanced Usage)](#高级用法-advanced-usage)
- [故障排除 (Troubleshooting)](#故障排除-troubleshooting)
- [示例 (Examples)](#示例-examples)

---

## 简介 (Introduction)

验证流水线 (ValidationPipeline) 是一个自动化工具，用于批量测试训练好的多无人机围捕模型并生成标准化的结果报告。

### 主要功能 (Key Features)

- ✅ **自动化验证**: 批量运行多个验证回合
- ✅ **灵活配置**: 支持 YAML 和 JSON 格式的配置文件
- ✅ **可视化输出**: 自动生成图片和视频
- ✅ **详细报告**: 生成包含关键指标的日志和报告
- ✅ **可重复性**: 支持随机种子设置，确保结果可重复
- ✅ **中文支持**: 完整的中文界面和报告

---

## 快速开始 (Quick Start)

### 1. 准备配置文件

复制示例配置文件并修改模型路径:

```bash
# 复制 YAML 配置文件
cp validation_config_example.yaml my_validation.yaml

# 或复制 JSON 配置文件
cp validation_config_example.json my_validation.json
```

编辑配置文件，修改模型路径:

```yaml
model:
  path: "model/your_model_directory"  # 修改为你的模型路径
```

### 2. 运行验证

使用 Python 运行验证流水线:

```python
from src.validation_pipeline import ValidationPipeline

# 创建验证流水线实例
pipeline = ValidationPipeline('my_validation.yaml')

# 运行验证
pipeline.run_validation()
```

或者使用命令行:

```bash
python -c "from src.validation_pipeline import ValidationPipeline; pipeline = ValidationPipeline('my_validation.yaml'); pipeline.run_validation()"
```

### 3. 查看结果

验证完成后，结果将保存在 `output/{timestamp}_validation/` 目录中。

---

## 配置文件 (Configuration File)

### 支持的格式 (Supported Formats)

- **YAML**: `.yaml` 或 `.yml` 文件 (推荐)
- **JSON**: `.json` 文件

### 配置选项 (Configuration Options)

#### 必需字段 (Required Fields)

```yaml
model:
  path: "model/your_model"  # 模型路径 (必需)

validation:
  num_episodes: 10  # 验证回合数 (必需)
```

#### 可选字段 (Optional Fields)

```yaml
validation:
  max_steps: 150              # 每回合最大步数 (默认: 150)
  save_frame_interval: 5      # 帧保存间隔 (默认: 5)

environment:
  num_hunters: 6              # Hunter 数量 (默认: 6)
  num_targets: 2              # Target 数量 (默认: 2)
  num_obstacles: 5            # 障碍物数量 (默认: 5)
  env_length: 2.0             # 环境大小 (默认: 2.0)

random_seed: 42               # 随机种子 (默认: 42)

output:
  save_images: true           # 保存图片 (默认: true)
  save_video: true            # 生成视频 (默认: true)
  save_logs: true             # 保存日志 (默认: true)
  video_fps: 10               # 视频帧率 (默认: 10)
```

### 配置示例 (Configuration Examples)

#### 快速测试配置

```yaml
model:
  path: "model/test_model"
validation:
  num_episodes: 3
  max_steps: 100
  save_frame_interval: 10
random_seed: 42
```

#### 高质量视频配置

```yaml
model:
  path: "model/best_model"
validation:
  num_episodes: 10
  max_steps: 200
  save_frame_interval: 2
output:
  video_fps: 30
random_seed: 42
```

#### 大规模验证配置

```yaml
model:
  path: "model/production_model"
validation:
  num_episodes: 50
  max_steps: 150
environment:
  num_hunters: 8
  num_targets: 3
  num_obstacles: 10
random_seed: 42
```

---

## 运行验证 (Running Validation)

### 方法 1: Python 脚本

创建一个 Python 脚本 (例如 `run_validation.py`):

```python
from src.validation_pipeline import ValidationPipeline

# 创建验证流水线
pipeline = ValidationPipeline('my_validation.yaml')

# 运行验证
pipeline.run_validation()

print("验证完成!")
```

运行脚本:

```bash
python run_validation.py
```

### 方法 2: 命令行

直接在命令行运行:

```bash
python -c "from src.validation_pipeline import ValidationPipeline; ValidationPipeline('my_validation.yaml').run_validation()"
```

### 方法 3: 交互式 Python

在 Python 交互式环境中:

```python
>>> from src.validation_pipeline import ValidationPipeline
>>> pipeline = ValidationPipeline('my_validation.yaml')
>>> pipeline.run_validation()
```

---

## 输出结果 (Output Results)

### 输出目录结构

验证完成后，结果保存在以下目录结构中:

```
output/
└── {timestamp}_validation/
    ├── images/              # 验证过程图片
    │   ├── episode_0/
    │   │   ├── step_0000.png
    │   │   ├── step_0005.png
    │   │   └── ...
    │   ├── episode_1/
    │   └── ...
    ├── videos/              # 验证视频
    │   ├── episode_0.mp4
    │   ├── episode_1.mp4
    │   └── ...
    ├── logs/                # 日志文件
    │   ├── validation_metrics.txt
    │   └── validation_report.md
    └── config/              # 配置备份
        └── validation_config.json
```

### 输出文件说明

#### 1. 图片 (Images)

- **位置**: `images/episode_{N}/`
- **格式**: PNG
- **命名**: `step_{XXXX}.png`
- **内容**: 每个时间步的环境状态可视化

#### 2. 视频 (Videos)

- **位置**: `videos/`
- **格式**: MP4
- **命名**: `episode_{N}.mp4`
- **内容**: 完整回合的动画视频

#### 3. 指标日志 (Metrics Log)

- **位置**: `logs/validation_metrics.txt`
- **格式**: 纯文本
- **内容**: 
  - 配置信息
  - 每回合详细指标
  - 汇总统计

示例内容:

```
================================================================================
验证指标报告
================================================================================

配置信息:
  模型路径: model/20241223_133158_score_550
  随机种子: 42
  验证回合数: 10
  最大步数: 150

每回合详细指标:

回合 1:
  总步数: 87
  捕获成功: 是
  平均Hunter奖励: 12.3456
  平均Target奖励: -8.7654

...

汇总统计:
  捕获成功率: 80.00% (8/10)
  平均步数: 92.50
  平均Hunter奖励: 11.2345
  平均Target奖励: -9.1234
```

#### 4. 验证报告 (Validation Report)

- **位置**: `logs/validation_report.md`
- **格式**: Markdown
- **内容**:
  - 配置信息
  - 汇总统计
  - 每回合详情表格
  - 输出文件说明

#### 5. 配置备份 (Config Backup)

- **位置**: `config/validation_config.json`
- **格式**: JSON
- **内容**: 验证使用的完整配置参数

---

## 高级用法 (Advanced Usage)

### 自定义验证流程

你可以继承 `ValidationPipeline` 类来自定义验证流程:

```python
from src.validation_pipeline import ValidationPipeline

class CustomValidationPipeline(ValidationPipeline):
    def _run_single_episode(self, episode, max_steps):
        # 自定义单回合验证逻辑
        metrics = super()._run_single_episode(episode, max_steps)
        
        # 添加自定义指标
        metrics['custom_metric'] = self._compute_custom_metric()
        
        return metrics
    
    def _compute_custom_metric(self):
        # 计算自定义指标
        return 0.0

# 使用自定义流水线
pipeline = CustomValidationPipeline('my_validation.yaml')
pipeline.run_validation()
```

### 批量验证多个模型

创建脚本批量验证多个模型:

```python
from src.validation_pipeline import ValidationPipeline
import os

# 模型列表
models = [
    "model/model_epoch_100",
    "model/model_epoch_200",
    "model/model_epoch_300"
]

# 批量验证
for model_path in models:
    print(f"\n验证模型: {model_path}")
    
    # 创建临时配置
    config = {
        'model': {'path': model_path},
        'validation': {'num_episodes': 10},
        'random_seed': 42
    }
    
    # 保存临时配置
    import json
    temp_config_path = 'temp_validation.json'
    with open(temp_config_path, 'w') as f:
        json.dump(config, f)
    
    # 运行验证
    pipeline = ValidationPipeline(temp_config_path)
    pipeline.run_validation()
    
    # 清理临时配置
    os.remove(temp_config_path)

print("\n所有模型验证完成!")
```

### 程序化访问验证结果

```python
from src.validation_pipeline import ValidationPipeline

# 运行验证
pipeline = ValidationPipeline('my_validation.yaml')
pipeline.run_validation()

# 访问验证指标
for metrics in pipeline.episode_metrics:
    episode = metrics['episode']
    steps = metrics['total_steps']
    success = metrics['capture_success']
    
    print(f"回合 {episode}: {steps} 步, 成功={success}")

# 计算自定义统计
success_rate = sum(1 for m in pipeline.episode_metrics if m['capture_success']) / len(pipeline.episode_metrics)
print(f"总体成功率: {success_rate:.2%}")
```

---

## 故障排除 (Troubleshooting)

### 常见问题 (Common Issues)

#### 1. 配置文件解析错误

**错误信息**: `ValueError: 配置文件解析失败`

**解决方案**:
- 检查 YAML/JSON 语法是否正确
- 确保缩进正确 (YAML 对缩进敏感)
- 使用在线 YAML/JSON 验证器检查格式

#### 2. 模型路径不存在

**错误信息**: `FileNotFoundError: 模型路径不存在`

**解决方案**:
- 检查配置文件中的模型路径是否正确
- 确保路径相对于项目根目录
- 验证模型目录包含必需的文件 (hunter_*/actor.pth, target_*/actor.pth)

#### 3. 缺少必需字段

**错误信息**: `ValueError: 配置文件缺少必需字段`

**解决方案**:
- 确保配置文件包含 `model.path` 和 `validation.num_episodes`
- 参考 `validation_config_example.yaml` 示例

#### 4. 中文显示乱码

**问题**: 生成的图片或报告中中文显示为方块

**解决方案**:
- 确保系统安装了宋体 (SimSun) 字体
- Windows: 通常已预装
- Linux: `sudo apt-get install fonts-wqy-zenhei`
- macOS: 下载并安装宋体字体

#### 5. 视频生成失败

**错误信息**: `警告: 无法读取图片` 或视频文件损坏

**解决方案**:
- 确保安装了 OpenCV: `pip install opencv-python`
- 检查图片是否正确生成
- 尝试降低 `video_fps` 参数

#### 6. 内存不足

**问题**: 验证过程中程序崩溃或系统变慢

**解决方案**:
- 减少 `num_episodes`
- 增加 `save_frame_interval` (减少保存的图片数量)
- 设置 `save_images: false` 只生成日志
- 分批运行验证

---

## 示例 (Examples)

### 示例 1: 基础验证

```yaml
# basic_validation.yaml
model:
  path: "model/my_model"
validation:
  num_episodes: 5
  max_steps: 100
random_seed: 42
```

```bash
python -c "from src.validation_pipeline import ValidationPipeline; ValidationPipeline('basic_validation.yaml').run_validation()"
```

### 示例 2: 高质量视频生成

```yaml
# high_quality_video.yaml
model:
  path: "model/best_model"
validation:
  num_episodes: 3
  max_steps: 200
  save_frame_interval: 1  # 保存每一帧
output:
  video_fps: 30  # 高帧率
random_seed: 42
```

### 示例 3: 快速测试 (不保存图片)

```yaml
# quick_test.yaml
model:
  path: "model/test_model"
validation:
  num_episodes: 2
  max_steps: 50
output:
  save_images: false  # 不保存图片
  save_video: false   # 不生成视频
  save_logs: true     # 只保存日志
random_seed: 42
```

### 示例 4: 大规模环境验证

```yaml
# large_scale.yaml
model:
  path: "model/large_model"
validation:
  num_episodes: 20
  max_steps: 200
environment:
  num_hunters: 10
  num_targets: 4
  num_obstacles: 15
  env_length: 3.0
random_seed: 42
```

### 示例 5: 对比不同随机种子

```python
# compare_seeds.py
from src.validation_pipeline import ValidationPipeline
import json

seeds = [42, 123, 456, 789]
results = {}

for seed in seeds:
    print(f"\n测试随机种子: {seed}")
    
    config = {
        'model': {'path': 'model/my_model'},
        'validation': {'num_episodes': 10},
        'random_seed': seed,
        'output': {'save_images': False, 'save_video': False}
    }
    
    config_path = f'temp_seed_{seed}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    pipeline = ValidationPipeline(config_path)
    pipeline.run_validation()
    
    # 记录结果
    success_rate = sum(1 for m in pipeline.episode_metrics if m['capture_success']) / len(pipeline.episode_metrics)
    results[seed] = success_rate
    
    print(f"种子 {seed} 成功率: {success_rate:.2%}")

print("\n汇总结果:")
for seed, rate in results.items():
    print(f"  种子 {seed}: {rate:.2%}")
```

---

## 依赖项 (Dependencies)

确保安装了以下 Python 包:

```bash
pip install numpy torch matplotlib pyyaml opencv-python
```

或使用 requirements.txt:

```bash
pip install -r requirements.txt
```

---

## 许可证 (License)

本验证流水线是多无人机围捕系统的一部分。

---

## 联系方式 (Contact)

如有问题或建议，请联系项目维护者。

---

## 更新日志 (Changelog)

### v1.0.0 (2024-12-23)
- ✅ 初始版本发布
- ✅ 支持 YAML 和 JSON 配置
- ✅ 自动生成图片、视频和报告
- ✅ 完整的中文支持
- ✅ 可配置的环境参数
- ✅ 随机种子支持

---

**祝验证顺利! (Happy Validating!)**
