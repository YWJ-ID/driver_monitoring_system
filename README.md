# 驾驶员监控系统 (Driver Monitoring System)

这是一个旨在监控驾驶员状态和行为的项目，可以实时检测司机的各种不良驾驶行为，如疲劳驾驶、分心驾驶等。

## 项目功能

### 🚨 可识别的司机不良行为

#### 1. **分心驾驶行为**
- **打电话 (Phone Call)** - 检测司机是否在驾驶过程中使用手机打电话
- **发短信/使用手机 (Texting)** - 检测司机是否在发短信或操作手机
- **吸烟** - 检测司机是否在吸烟

#### 2. **疲劳驾驶行为**
- **闭眼 (Eye Closed)** - 检测司机眼睛是否长时间闭合，这是疲劳驾驶的重要预警信号
- **打哈欠 (Yawning)** - 检测司机是否在打哈欠，表明疲劳状态

#### 3. **注意力分散行为**
- **视线方向检测**：
  - 向左看 (Gazing Left) - 检测司机是否频繁向左看，可能分散注意力
  - 向右看 (Gazing Right) - 检测司机是否频繁向右看，可能分散注意力
  - 向前看 (Gazing Center) - 正常的前方注视状态

## 技术实现详解

### 🏗️ 核心技术栈

本项目采用**多模态深度学习**技术，结合了以下关键技术：

#### **计算机视觉技术**
- **OpenCV**: 用于图像处理和视频流处理
- **MediaPipe**: Google开发的面部特征点检测和追踪
- **YOLOv8**: 用于目标检测，专门检测手机设备和香烟

#### **深度学习框架**
- **TensorFlow**: 用于行为分类模型的训练和推理
- **PyTorch**: 用于YOLOv8目标检测模型

### 🏛️ 系统架构设计

系统采用**双模块架构**设计：

```
┌─────────────────┐    ┌─────────────────┐
│   面部追踪模块    │    │   行为检测模块    │
│  (MediaPipe)    │    │   (MobileNet)   │
│                 │    │                 │
│ • 眼部状态检测    │    │ • 特征提取       │
│ • 视线方向追踪    │    │ • 行为分类       │
│ • 打哈欠检测     ｜    ｜• YOLO手机检测   ｜
| • 手部吸烟动作    │    │ • YOLO香烟检测   │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│             多模态融合决策系统             │
│           (状态融合与行为识别)             │
└─────────────────────────────────────────┘
                    ↓
                输出监控结果
```

### 🔍 面部追踪模块技术实现

#### **MediaPipe面部网格**
```python
# 面部特征点检测
self.fm = FaceMesh()  # 468个面部特征点
```

**技术特点：**
- 实时检测468个面部特征点
- 支持多人脸同时检测
- 高精度面部轮廓和关键点定位

#### **眼部状态检测算法**
```python
# 眼睛闭合检测算法
def eye_closed(self, threshold=conf.EYE_CLOSED):
    return self.eye_veti_to_hori < threshold
```

**技术原理：**
- 计算眼睛垂直距离与水平距离的比值
- 当比值小于阈值时判定为闭眼
- 连续多帧检测提高准确性

#### **视线方向检测算法**
```python
# 视线方向判断
def gaze_left(self, threshold=conf.GAZE_LEFT):
    return self.iris_relative_to_eye[0] < threshold
```

**技术原理：**
- 计算虹膜相对于眼睛的位置比例
- 通过虹膜位置判断视线方向（左/中/右）
- 左右眼协同判断提高准确性

#### **打哈欠检测**
```python
# 嘴部状态检测
def mouth_open(self):
    # 通过嘴部特征点计算开口程度
```

### 🤖 行为检测模块技术实现

#### **MobileNet架构**
```python
class MobileNet(Sequential):
    def __init__(self, input_shape=(224,224,3), num_classes=2):
        self.base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False)
```

**技术特点：**
- 基于预训练的MobileNetV2
- 轻量化设计，适合实时推理
- 支持迁移学习，提高训练效率

#### **YOLOv8目标检测**
```python
yolo_result = yolo_model(rgb_image, classes=[67, 74], conf=0.25)  # 检测手机和香烟
```

**技术特点：**
- 专门检测手机和香烟（COCO数据集类别67，74）
- 实时目标检测能力
- 高精度设备识别

### 🔄 多模态融合决策

#### **状态融合算法**
```python
def infer_one_frame(image, model, yolo_model, facial_tracker):
    # 面部状态检测
    eyes_status = facial_tracker.eyes_status
    yawn_status = facial_tracker.yawn_status
    
    # 行为分类
    y = model.predict(rgb_image)
    result = np.argmax(y, axis=1)
    
    # 多模态决策
    if result[0] == 0 and yolo_result.xyxy[0].shape[0] > 0:
        action = 'phone call'  # 检测到手机且分类为打电话
    if result[0] == 1 and eyes_status == 'eye closed':
        action = 'texting'    # 分类为发短信且检测到闭眼
```

**决策逻辑：**
1. **打电话检测**: MobileNet行为分类 + YOLO手机检测
2. **发短信检测**: MobileNet行为分类 + 眼部状态验证
3. **疲劳检测**: 眼部状态 + 打哈欠检测
4. **吸烟检测**: 手部吸烟动作 + YOLO香烟检测

### 📊 数据集与训练

#### **DMD数据集**
- 5个驾驶员，每人4个不同场景
- 包含打电话和发短信行为
- 支持监督学习训练

#### **数据预处理**
```python
def preprocess_image(image, size):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size)
    return image
```


## 环境要求

### 常规依赖包
```
python==3.8
tensorflow==2.8.0
torch==1.11.0
opencv-python==4.5.5
mediapipe==0.8.9.1
matplotlib==3.5.1
numpy==1.22.3
scikit-learn==1.0.2
```

### Apple Silicon Mac 特殊要求
如果您使用的是 Apple Silicon Mac (M1/M2/M3 芯片)，请使用以下依赖包：

```
python==3.8
tensorflow-macos==2.13.0
tensorflow-metal==1.0.0
torch==1.11.0
torchvision==0.12.0
opencv-python==4.5.5.64
mediapipe==0.10.5
matplotlib==3.5.1
numpy==1.24.3
scikit-learn==1.0.2
pandas==2.0.3
seaborn==0.13.2
```

**注意**：在 Apple Silicon Mac 上，必须使用 `tensorflow-macos` 而不是标准的 `tensorflow`，以确保最佳性能和兼容性。

## 使用方法

```bash
# 克隆项目
$ git clone https://gitlab.sjfood.us/neil/driver_monitoring.git
$ cd driver_monitoring

# 运行驾驶员监控系统
$ python3 dms.py --checkpoint models/model_split.h5 --video <视频路径> 
                                                    --webcam <摄像头ID> # 或

# 仅运行面部追踪功能
$ python3 facial.py
```

## 数据集

行为检测模型使用的训练数据集是 [DMD](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset) (Driver Monitoring Dataset)。

## 演示效果

<p align="center">
  <img src="https://user-images.githubusercontent.com/62132206/158055802-8e1146f8-32ef-4ae4-967a-eb79ac42e172.gif?raw=true">
  <img src="https://user-images.githubusercontent.com/62132206/158055799-22effa40-89d2-46da-a317-d58ea3e186b5.gif?raw=true">
</p>

## 项目结构

```
driver_monitoring/
├── dms.py                    # 主程序入口
├── net.py                    # MobileNet模型定义
├── train.py                  # 模型训练脚本
├── facial.py                 # 面部追踪独立运行
├── dms_utils/                # 工具函数
│   └── dms_utils.py
├── facial_tracking/          # 面部追踪模块
│   ├── facialTracking.py     # 面部追踪主类
│   ├── faceMesh.py          # MediaPipe面部网格
│   ├── eye.py               # 眼部状态检测
│   ├── iris.py              # 虹膜追踪
│   ├── lips.py              # 嘴部状态检测
│   └── conf.py              # 配置文件
├── models/                   # 预训练模型
│   ├── model_random.h5
│   └── model_split.h5
└── images/                   # 示例图片
```

