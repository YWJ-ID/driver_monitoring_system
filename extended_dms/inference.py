#!/usr/bin/env python3
"""
扩展驾驶员监控系统推理脚本
使用训练好的模型进行实时预测
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import json
import argparse
from ultralytics import YOLO

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import ExtendedDriverMonitor

class ExtendedDriverInference:
    """扩展驾驶员监控推理类"""
    
    def __init__(self, model_path, class_mapping_path=None, img_size=(224, 224)):
        self.img_size = img_size
        self.model = None
        self.class_names = {}
        
        # 加载模型
        self.load_model(model_path, class_mapping_path)
        
        # 初始化 YOLO 模型用于设备检测
        # 使用 torch.serialization.add_safe_globals 来处理 PyTorch 2.6 的 weights_only 限制
        import torch
        try:
            # PyTorch 2.6+ 需要添加安全全局变量
            from ultralytics.nn.tasks import DetectionModel
            from torch.nn.modules.container import Sequential
            torch.serialization.add_safe_globals([DetectionModel, Sequential, type])
        except (AttributeError, TypeError, ImportError):
            pass  # 旧版本 PyTorch 不需要
                
        self.yolo_model = YOLO('yolov8s.pt')
        
        # 行为类别颜色映射 - 合并版本
        self.action_colors = {
            'safe_driving': (0, 255, 0),      # 绿色 - 安全
            'texting': (0, 0, 255),           # 红色 - 危险
            'phone_call': (0, 0, 255),        # 红色 - 危险
            'radio_operation': (0, 165, 255), # 橙色 - 警告
            'drinking': (0, 165, 255),        # 橙色 - 警告
            'looking_back': (0, 165, 255),    # 橙色 - 警告
            'makeup': (0, 165, 255),          # 橙色 - 警告
            'passenger_talk': (0, 165, 255)   # 橙色 - 警告
        }
        
        # 行为风险等级 - 合并版本
        self.risk_levels = {
            'safe_driving': '低风险',
            'texting': '高风险',
            'phone_call': '高风险',
            'radio_operation': '中风险',
            'drinking': '中风险',
            'looking_back': '中风险',
            'makeup': '中风险',
            'passenger_talk': '中风险'
        }
    
    def load_model(self, model_path, class_mapping_path=None):
        """加载训练好的模型"""
        print(f"正在加载模型: {model_path}")
        
        # 加载模型
        self.model = ExtendedDriverMonitor()
        self.model.load_model(model_path)
        
        # 加载类别映射
        if class_mapping_path and os.path.exists(class_mapping_path):
            with open(class_mapping_path, 'r', encoding='utf-8') as f:
                class_mapping = json.load(f)
                self.class_names = {int(k): v for k, v in class_mapping.items()}
        else:
            # 使用默认类别映射
            self.class_names = self.model.class_names
        
        print(f"模型加载完成，支持 {len(self.class_names)} 种行为类别")
        for class_id, class_name in self.class_names.items():
            print(f"  {class_id}: {class_name}")
    
    def preprocess_image(self, image):
        """预处理图像"""
        # 调整图像大小
        image_resized = cv2.resize(image, self.img_size)
        
        # 转换为RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # 归一化
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # 添加批次维度
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def detect_devices(self, image):
        """使用YOLO检测手机等设备"""
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # YOLO检测，只检测手机(67)和香烟(74)
        results = self.yolo_model(image_rgb, classes=[67, 74], conf=0.25)
        
        devices = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                device_type = "手机" if class_id == 67 else "香烟"
                devices.append({
                    'type': device_type,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
        
        return devices
    
    def predict_behavior(self, image):
        """预测驾驶行为"""
        # 预处理图像
        image_batch = self.preprocess_image(image)
        
        # 模型预测
        predictions = self.model.model.predict(image_batch, verbose=0)
        
        # 获取预测结果
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions, axis=1)[0]
        
        # 获取类别名称
        class_name = self.class_names.get(predicted_class, f"未知类别_{predicted_class}")
        
        return {
            'class_id': predicted_class,
            'class_name': class_name,
            'confidence': confidence,
            'predictions': predictions[0]
        }
    
    def analyze_frame(self, image):
        """分析单帧图像"""
        # 行为预测
        behavior_result = self.predict_behavior(image)
        
        # 设备检测
        devices = self.detect_devices(image)
        
        # 综合分析
        analysis_result = {
            'behavior': behavior_result,
            'devices': devices,
            'risk_level': self.risk_levels.get(behavior_result['class_name'], '未知'),
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
        }
        
        return analysis_result
    
    def draw_results(self, image, analysis_result):
        """在图像上绘制结果"""
        # 获取行为信息
        behavior = analysis_result['behavior']
        devices = analysis_result['devices']
        risk_level = analysis_result['risk_level']
        
        # 获取颜色
        action_color = self.action_colors.get(behavior['class_name'], (255, 255, 255))
        
        # 绘制行为信息
        behavior_text = f"行为: {behavior['class_name']}"
        confidence_text = f"置信度: {behavior['confidence']:.3f}"
        risk_text = f"风险等级: {risk_level}"
        
        # 绘制文本
        cv2.putText(image, behavior_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, action_color, 2, cv2.LINE_AA)
        cv2.putText(image, confidence_text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, action_color, 2, cv2.LINE_AA)
        cv2.putText(image, risk_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, action_color, 2, cv2.LINE_AA)
        
        # 绘制设备检测结果
        for i, device in enumerate(devices):
            x1, y1, x2, y2 = device['bbox']
            device_text = f"{device['type']}: {device['confidence']:.2f}"
            
            # 绘制边界框
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 绘制标签
            cv2.putText(image, device_text, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 绘制警告框（如果是高风险行为）
        if risk_level == '高风险':
            cv2.rectangle(image, (10, 10), (image.shape[1]-10, image.shape[0]-10), 
                         (0, 0, 255), 3)
            cv2.putText(image, "警告: 高风险驾驶行为!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        
        return image
    
    def process_video(self, video_path, output_path=None, show_display=True):
        """处理视频文件"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 设置输出视频
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        print("开始处理视频...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 分析帧
            analysis_result = self.analyze_frame(frame)
            
            # 绘制结果
            frame_with_results = self.draw_results(frame, analysis_result)
            
            # 显示帧
            if show_display:
                cv2.imshow('扩展驾驶员监控', frame_with_results)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 保存帧
            if output_path:
                out.write(frame_with_results)
            
            # 打印进度
            if frame_count % 30 == 0:
                print(f"已处理 {frame_count} 帧")
        
        # 清理
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"视频处理完成，共处理 {frame_count} 帧")
    
    def process_webcam(self, cam_id=0):
        """处理摄像头实时流"""
        cap = cv2.VideoCapture(cam_id)
        
        if not cap.isOpened():
            print(f"无法打开摄像头: {cam_id}")
            return
        
        print("开始实时监控，按 'q' 退出...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
            
            # 分析帧
            analysis_result = self.analyze_frame(frame)
            
            # 绘制结果
            frame_with_results = self.draw_results(frame, analysis_result)
            
            # 显示帧
            cv2.imshow('扩展驾驶员监控 - 实时', frame_with_results)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("实时监控结束")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='扩展驾驶员监控推理')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--class_mapping', type=str, default=None,
                       help='类别映射文件路径')
    parser.add_argument('--video', type=str, default=None,
                       help='输入视频文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频文件路径')
    parser.add_argument('--webcam', type=int, default=None,
                       help='摄像头ID')
    parser.add_argument('--image', type=str, default=None,
                       help='单张图像路径')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 创建推理器
    inference = ExtendedDriverInference(
        model_path=args.model_path,
        class_mapping_path=args.class_mapping
    )
    
    # 根据输入类型进行处理
    if args.image:
        # 处理单张图像
        image = cv2.imread(args.image)
        if image is None:
            print(f"无法读取图像: {args.image}")
            return
        
        analysis_result = inference.analyze_frame(image)
        image_with_results = inference.draw_results(image, analysis_result)
        
        # 显示结果
        cv2.imshow('扩展驾驶员监控', image_with_results)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 打印分析结果
        print("分析结果:")
        print(f"  行为: {analysis_result['behavior']['class_name']}")
        print(f"  置信度: {analysis_result['behavior']['confidence']:.3f}")
        print(f"  风险等级: {analysis_result['risk_level']}")
        print(f"  检测到的设备: {len(analysis_result['devices'])} 个")
        
    elif args.video:
        # 处理视频
        inference.process_video(args.video, args.output)
        
    elif args.webcam is not None:
        # 处理摄像头
        inference.process_webcam(args.webcam)
        
    else:
        print("请指定输入类型: --image, --video, 或 --webcam")

if __name__ == "__main__":
    main() 