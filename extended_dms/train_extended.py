#!/usr/bin/env python3
"""
扩展驾驶员监控系统训练脚本
使用StateFarm数据集训练MobileNetV2模型
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import StateFarmDataLoader
from model import ExtendedDriverMonitor

def create_directories():
    """创建必要的目录"""
    directories = [
        'extended_dms/results',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.show()

def plot_class_accuracy(y_true, y_pred, class_names, save_path=None):
    """绘制各类别准确率"""
    # 计算各类别准确率
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), class_accuracy, color='lightcoral', alpha=0.7)
    
    # 添加数值标签
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('行为类别')
    plt.ylabel('准确率')
    plt.title('各类别准确率')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"类别准确率图已保存到: {save_path}")
    
    plt.show()

def evaluate_model(model, test_dataset, data_loader, save_dir):
    """评估模型性能"""
    print("开始模型评估...")
    
    # 收集预测结果
    y_true = []
    y_pred = []
    
    for images, labels in test_dataset:
        predictions = model.model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算整体准确率
    accuracy = np.mean(y_true == y_pred)
    print(f"整体准确率: {accuracy:.4f}")
    
    # 分类报告
    class_names = data_loader.get_all_class_names()
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 绘制混淆矩阵
    confusion_matrix_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, class_names, confusion_matrix_path)
    
    # 绘制类别准确率
    class_accuracy_path = os.path.join(save_dir, 'class_accuracy.png')
    plot_class_accuracy(y_true, y_pred, class_names, class_accuracy_path)
    
    return accuracy, y_true, y_pred

def save_training_results(model, data_loader, save_dir):
    """保存训练结果"""
    # 保存模型
    model_path = os.path.join(save_dir, 'extended_driver_monitor_model.h5')
    model.save_model(model_path)
    
    # 保存类别映射
    class_mapping = {}
    for i, class_name in enumerate(data_loader.get_all_class_names()):
        class_mapping[i] = class_name
    
    import json
    mapping_path = os.path.join(save_dir, 'class_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"类别映射已保存到: {mapping_path}")
    
    # 保存训练历史
    if model.history:
        history_path = os.path.join(save_dir, 'training_history.png')
        model.plot_training_history(history_path)

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='训练扩展驾驶员监控模型')
    parser.add_argument('--data_path', type=str, 
                       default='datasets/state-farm-distracted-driver-detection',
                       help='StateFarm数据集路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮次')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--save_dir', type=str, default='extended_dms/results', 
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 创建带时间戳和训练参数的结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_name = f"train_{timestamp}_e{args.epochs}_bs{args.batch_size}_lr{args.learning_rate:.0e}"
    timestamped_save_dir = os.path.join(args.save_dir, train_name)
    
    print("🚗 扩展驾驶员监控系统训练")
    print("=" * 50)
    print(f"数据集路径: {args.data_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮次: {args.epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"图像尺寸: {args.img_size}x{args.img_size}")
    print(f"训练结果将保存到: {timestamped_save_dir}")
    print("=" * 50)
    
    # 创建目录
    create_directories()
    os.makedirs(timestamped_save_dir, exist_ok=True)
    print(f"创建训练结果目录: {timestamped_save_dir}")
    
    # 更新保存目录
    args.save_dir = timestamped_save_dir
    
    # 检查数据集是否存在
    if not os.path.exists(args.data_path):
        print(f"错误: 数据集路径不存在: {args.data_path}")
        print("请先下载StateFarm数据集")
        return
    
    try:
        # 1. 加载数据
        print("\n1. 加载数据集...")
        data_loader = StateFarmDataLoader(
            data_path=args.data_path,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size
        )
        
        # 尝试从CSV文件加载
        csv_path = os.path.join(args.data_path, "driver_imgs_list.csv")
        if os.path.exists(csv_path):
            image_paths, labels = data_loader.load_data_from_csv(csv_path)
        else:
            image_paths, labels = data_loader.load_data_from_directories()
        
        # 分割数据集
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data(
            image_paths, labels
        )
        
        # 创建数据集
        train_dataset = data_loader.create_dataset(X_train, y_train, augment=True)
        val_dataset = data_loader.create_dataset(X_val, y_val, augment=False)
        test_dataset = data_loader.create_dataset(X_test, y_test, augment=False)
        
        # 2. 构建模型
        print("\n2. 构建模型...")
        model = ExtendedDriverMonitor(
            num_classes=data_loader.num_classes,
            img_size=(args.img_size, args.img_size),
            learning_rate=args.learning_rate
        )
        model.build_model(dropout_rate=args.dropout_rate)
        
        # 3. 创建回调函数
        print("\n3. 设置训练回调...")
        model_save_path = os.path.join(args.save_dir, 'best_model.h5')
        callbacks = model.create_callbacks(model_save_path, patience=args.patience)
        
        # 4. 训练模型
        print("\n4. 开始训练...")
        history = model.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=args.epochs,
            callbacks=callbacks
        )
        
        # 5. 评估模型
        print("\n5. 评估模型...")
        accuracy, y_true, y_pred = evaluate_model(
            model, test_dataset, data_loader, args.save_dir
        )
        
        # 6. 保存结果
        print("\n6. 保存训练结果...")
        save_training_results(model, data_loader, args.save_dir)
        
        # 7. 输出最终结果
        print("\n" + "=" * 50)
        print("🎉 训练完成！")
        print(f"最终测试准确率: {accuracy:.4f}")
        print(f"模型已保存到: {model_save_path}")
        print(f"训练结果已保存到: {args.save_dir}")
        print(f"目录名称: {train_name}")
        print("=" * 50)
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 