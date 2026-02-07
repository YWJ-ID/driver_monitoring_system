# 扩展驾驶员监控系统

基于StateFarm数据集和MobileNetV2的扩展驾驶员监控系统，支持10种不良驾驶行为的识别。

## 🚗 功能特性

### 支持的行为类别
1. **安全驾驶** (safe_driving) - 正常驾驶状态
2. **右手发短信** (texting_right) - 使用右手操作手机发短信
3. **右手打电话** (phone_call_right) - 使用右手接打电话
4. **左手发短信** (texting_left) - 使用左手操作手机发短信
5. **左手打电话** (phone_call_left) - 使用左手接打电话
6. **操作收音机** (radio_operation) - 操作车载收音机
7. **喝水** (drinking) - 在驾驶过程中喝水
8. **向后看** (looking_back) - 转头向后看
9. **化妆** (makeup) - 在驾驶过程中化妆
10. **与乘客交谈** (passenger_talk) - 与车内乘客交谈

### 技术特点
- **多模态检测**: 结合行为分类和设备检测
- **实时处理**: 支持实时视频流处理
- **风险分级**: 自动评估驾驶行为风险等级
- **可视化**: 丰富的可视化界面和结果展示

## 📁 项目结构

```
extended_dms/
├── data_loader.py          # 数据加载器
├── model.py               # 模型定义
├── train_extended.py      # 训练脚本
├── inference.py           # 推理脚本
├── requirements.txt       # 依赖包列表
├── README.md             # 使用说明
├── models/               # 模型保存目录
├── results/              # 训练结果目录
├── logs/                 # 日志目录
└── plots/                # 图表保存目录
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖包
pip install -r requirements.txt

# 或者使用conda
conda create -n extended_dms python=3.8
conda activate extended_dms
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
# 运行下载脚本
python download_datasets.py

# 或手动下载StateFarm数据集
# 访问: https://www.kaggle.com/c/state-farm-distracted-driver-detection
# 下载并解压到 datasets/state-farm-distracted-driver-detection/
```

### 3. 训练模型

```bash
# 基本训练
python train_extended.py --data_path datasets/state-farm-distracted-driver-detection

# 自定义参数训练
python train_extended.py \
    --data_path datasets/state-farm-distracted-driver-detection \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 1e-3 \
    --img_size 224 \
    --dropout_rate 0.5 \
    --patience 15
```

### 4. 使用训练好的模型

```bash
# 处理单张图像
python inference.py \
    --model_path extended_dms/results/extended_driver_monitor_model.h5 \
    --class_mapping extended_dms/results/class_mapping.json \
    --image path/to/image.jpg

# 处理视频文件
python inference.py \
    --model_path extended_dms/results/extended_driver_monitor_model.h5 \
    --class_mapping extended_dms/results/class_mapping.json \
    --video path/to/video.mp4 \
    --output output_video.mp4

# 实时摄像头监控
python inference.py \
    --model_path extended_dms/results/extended_driver_monitor_model.h5 \
    --class_mapping extended_dms/results/class_mapping.json \
    --webcam 0
```

## 📊 训练参数说明

### 数据参数
- `--data_path`: StateFarm数据集路径
- `--batch_size`: 批次大小 (默认: 32)
- `--img_size`: 图像尺寸 (默认: 224)

### 模型参数
- `--learning_rate`: 学习率 (默认: 1e-3)
- `--dropout_rate`: Dropout率 (默认: 0.5)
- `--epochs`: 训练轮次 (默认: 30)
- `--patience`: 早停耐心值 (默认: 10)

### 输出参数
- `--save_dir`: 结果保存目录 (默认: extended_dms/results)

## 🎯 模型架构

### MobileNetV2 + 自定义分类头
```
输入图像 (224x224x3)
    ↓
MobileNetV2 (预训练)
    ↓
全局平均池化
    ↓
全连接层 (1024) + BatchNorm + Dropout
    ↓
全连接层 (512) + BatchNorm + Dropout
    ↓
全连接层 (256) + BatchNorm + Dropout
    ↓
输出层 (10) + Softmax
```

### 数据增强策略
- 随机水平翻转
- 随机亮度调整 (±20%)
- 随机对比度调整 (0.8-1.2)
- 随机饱和度调整 (0.8-1.2)

## 📈 性能评估

### 评估指标
- **准确率 (Accuracy)**: 整体分类准确率
- **精确率 (Precision)**: 每个类别的精确率
- **召回率 (Recall)**: 每个类别的召回率
- **F1分数**: 精确率和召回率的调和平均

### 可视化结果
- 训练历史曲线
- 混淆矩阵
- 各类别准确率柱状图
- 实时检测结果

## 🔧 高级功能

### 多模态融合
```python
# 行为分类 + 设备检测
behavior_result = model.predict_behavior(image)
devices = yolo_model.detect_devices(image)

# 综合分析
if behavior_result['class_name'] in ['texting_right', 'phone_call_right']:
    if devices:  # 检测到手机
        risk_level = '高风险'
    else:
        risk_level = '中风险'
```

### 风险等级评估
- **低风险**: 安全驾驶
- **中风险**: 操作收音机、喝水、向后看、化妆、与乘客交谈
- **高风险**: 使用手机发短信、打电话

### 实时警告系统
- 高风险行为自动触发警告框
- 颜色编码显示风险等级
- 置信度实时显示

## 🛠️ 自定义扩展

### 添加新的行为类别
1. 修改 `data_loader.py` 中的 `class_names` 字典
2. 更新 `model.py` 中的类别映射
3. 重新训练模型

### 调整检测阈值
```python
# 在 inference.py 中修改
confidence_threshold = 0.7  # 置信度阈值
risk_threshold = 0.8        # 风险阈值
```

### 集成到现有系统
```python
# 在原有的 dms.py 中集成
from extended_dms.inference import ExtendedDriverInference

# 创建推理器
inference = ExtendedDriverInference(
    model_path='extended_dms/results/extended_driver_monitor_model.h5'
)

# 在原有推理函数中使用
def infer_one_frame_extended(image, model, yolo_model, facial_tracker):
    # 原有的面部追踪逻辑
    facial_tracker.process_frame(image)
    
    # 扩展的行为检测
    analysis_result = inference.analyze_frame(image)
    
    # 显示结果
    image = inference.draw_results(image, analysis_result)
    
    return image
```

## ⚠️ 注意事项

1. **数据集要求**: 确保StateFarm数据集正确下载和解压
2. **GPU支持**: 建议使用GPU加速训练
3. **内存要求**: 训练时需要足够的内存和显存
4. **实时性能**: 推理时注意调整图像尺寸以平衡精度和速度
5. **隐私保护**: 确保遵守相关隐私法规

## 🐛 常见问题

### Q: 训练时出现内存不足错误
A: 减小batch_size或图像尺寸

### Q: 模型准确率不高
A: 尝试增加训练轮次、调整学习率或使用数据增强

### Q: 推理速度慢
A: 减小图像尺寸或使用更轻量的模型

### Q: 检测结果不准确
A: 检查数据集质量，调整置信度阈值

## 📚 参考资料

- [StateFarm数据集](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
- [MobileNetV2论文](https://arxiv.org/abs/1801.04381)
- [TensorFlow文档](https://www.tensorflow.org/)
- [OpenCV文档](https://opencv.org/)

## 📄 许可证

本项目基于MIT许可证开源。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！ 