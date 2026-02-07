import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class StateFarmDataLoader:
    """StateFarm分心驾驶数据集加载器"""
    
    def __init__(self, data_path, img_size=(224, 224), batch_size=32):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        
        # StateFarm数据集的行为类别 - 合并版本
        self.original_class_names = {
            'c0': 'safe_driving',      # 安全驾驶
            'c1': 'texting_right',     # 右手发短信
            'c2': 'phone_call_right',  # 右手打电话
            'c3': 'texting_left',      # 左手发短信
            'c4': 'phone_call_left',   # 左手打电话
            'c5': 'radio_operation',   # 操作收音机
            'c6': 'drinking',          # 喝水
            'c7': 'looking_back',      # 向后看
            'c8': 'makeup',            # 化妆
            'c9': 'passenger_talk'     # 与乘客交谈
        }
        
        # 合并后的类别映射
        self.merged_class_names = {
            0: 'safe_driving',      # 安全驾驶
            1: 'texting',           # 发短信 (合并 c1 + c3)
            2: 'phone_call',        # 打电话 (合并 c2 + c4)
            3: 'radio_operation',   # 操作收音机
            4: 'drinking',          # 喝水
            5: 'looking_back',      # 向后看
            6: 'makeup',            # 化妆
            7: 'passenger_talk'     # 与乘客交谈
        }
        
        # 原始类别到合并类别的映射
        self.merge_mapping = {
            'c0': 0,  # safe_driving
            'c1': 1,  # texting_right -> texting
            'c2': 2,  # phone_call_right -> phone_call
            'c3': 1,  # texting_left -> texting
            'c4': 2,  # phone_call_left -> phone_call
            'c5': 3,  # radio_operation
            'c6': 4,  # drinking
            'c7': 5,  # looking_back
            'c8': 6,  # makeup
            'c9': 7   # passenger_talk
        }
        
        self.num_classes = len(self.merged_class_names)
        
    def load_data_from_csv(self, csv_path):
        """从CSV文件加载数据"""
        print("正在加载StateFarm数据集...")
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        print(f"数据集大小: {len(df)} 张图像")
        
        # 统计各类别数量
        class_counts = df['classname'].value_counts()
        print("\n原始类别图像数量:")
        for class_name, count in class_counts.items():
            print(f"{self.original_class_names[class_name]}: {count}")
        
        # 创建图像路径和标签
        image_paths = []
        labels = []
        
        for _, row in df.iterrows():
            # 构建图像路径
            img_path = os.path.join(self.data_path, 'train', row['classname'], row['img'])
            
            # 检查文件是否存在
            if os.path.exists(img_path):
                image_paths.append(img_path)
                # 使用合并映射获取标签
                labels.append(self.merge_mapping[row['classname']])
        
        # 统计合并后的类别数量
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\n合并后的类别图像数量:")
        for label_id, count in zip(unique_labels, counts):
            print(f"{self.merged_class_names[label_id]}: {count}")
        
        print(f"\n有效图像数量: {len(image_paths)}")
        
        return np.array(image_paths), np.array(labels)
    
    def load_data_from_directories(self):
        """从目录结构加载数据"""
        print("正在从目录结构加载StateFarm数据集...")
        
        image_paths = []
        labels = []
        
        train_dir = os.path.join(self.data_path, 'train')
        
        for class_name in self.original_class_names.keys():
            class_dir = os.path.join(train_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            merged_class_id = self.merge_mapping[class_name]
            class_images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
            
            for img_name in class_images:
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(merged_class_id)
        
        print(f"总图像数量: {len(image_paths)}")
        
        # 统计合并后的类别数量
        unique, counts = np.unique(labels, return_counts=True)
        print("\n合并后的类别图像数量:")
        for class_id, count in zip(unique, counts):
            print(f"{self.merged_class_names[class_id]}: {count}")
        
        return np.array(image_paths), np.array(labels)
    
    def preprocess_image(self, image_path):
        """预处理单张图像"""
        # 加载图像
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        
        # 归一化到[0,1]
        img_array = img_array / 255.0
        
        return img_array
    
    def create_dataset(self, image_paths, labels, shuffle=True, augment=False):
        """创建TensorFlow数据集"""
        def load_and_preprocess(path, label):
            # 读取图像文件
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.img_size)
            img = tf.cast(img, tf.float32) / 255.0
            
            if augment:
                # 数据增强
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, 0.2)
                img = tf.image.random_contrast(img, 0.8, 1.2)
                img = tf.image.random_saturation(img, 0.8, 1.2)
                img = tf.clip_by_value(img, 0.0, 1.0)
            
            return img, label
        
        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def split_data(self, image_paths, labels, test_size=0.2, val_size=0.2, random_state=42):
        """分割数据集为训练、验证和测试集"""
        # 首先分割出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        
        # 从剩余数据中分割出验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"训练集: {len(X_train)} 张图像")
        print(f"验证集: {len(X_val)} 张图像")
        print(f"测试集: {len(X_test)} 张图像")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def plot_class_distribution(self, labels, title="类别分布"):
        """绘制类别分布图"""
        plt.figure(figsize=(12, 6))
        
        # 统计各类别数量
        unique, counts = np.unique(labels, return_counts=True)
        class_names = [self.merged_class_names[i] for i in unique]
        
        # 创建柱状图
        bars = plt.bar(range(len(class_names)), counts, color='skyblue', alpha=0.7)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('行为类别')
        plt.ylabel('图像数量')
        plt.title(title)
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def plot_sample_images(self, image_paths, labels, num_samples=8):
        """绘制样本图像"""
        plt.figure(figsize=(16, 8))
        
        for i in range(min(num_samples, len(image_paths))):
            plt.subplot(2, 4, i + 1)
            
            # 加载和显示图像
            img = self.preprocess_image(image_paths[i])
            plt.imshow(img)
            
            # 获取类别名称
            plt.title(f"{self.merged_class_names[labels[i]]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_all_class_names(self):
        """获取所有类别名称"""
        return [self.merged_class_names[i] for i in range(self.num_classes)]

def main():
    """测试数据加载器"""
    # 设置数据路径（需要根据实际情况修改）
    data_path = "datasets/state-farm-distracted-driver-detection"
    csv_path = os.path.join(data_path, "driver_imgs_list.csv")
    
    # 创建数据加载器
    data_loader = StateFarmDataLoader(data_path)
    
    # 检查CSV文件是否存在
    if os.path.exists(csv_path):
        print("使用CSV文件加载数据...")
        image_paths, labels = data_loader.load_data_from_csv(csv_path)
    else:
        print("CSV文件不存在，使用目录结构加载数据...")
        image_paths, labels = data_loader.load_data_from_directories()
    
    # 绘制类别分布
    data_loader.plot_class_distribution(labels)
    
    # 绘制样本图像
    data_loader.plot_sample_images(image_paths, labels)
    
    # 分割数据集
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data(
        image_paths, labels
    )
    
    # 创建数据集
    train_dataset = data_loader.create_dataset(X_train, y_train, augment=True)
    val_dataset = data_loader.create_dataset(X_val, y_val, augment=False)
    test_dataset = data_loader.create_dataset(X_test, y_test, augment=False)
    
    print("数据集创建完成！")
    
    return data_loader, train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    main() 