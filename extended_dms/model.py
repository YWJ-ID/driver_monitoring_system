import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

class ExtendedDriverMonitor:
    """扩展的驾驶员监控模型"""
    
    def __init__(self, num_classes=10, img_size=(224, 224), learning_rate=1e-3):
        self.num_classes = num_classes
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
        # 类别名称映射 - 合并版本
        self.class_names = {
            0: 'safe_driving',      # 安全驾驶
            1: 'texting',           # 发短信 (合并 右手+左手)
            2: 'phone_call',        # 打电话 (合并 右手+左手)
            3: 'radio_operation',   # 操作收音机
            4: 'drinking',          # 喝水
            5: 'looking_back',      # 向后看
            6: 'makeup',            # 化妆
            7: 'passenger_talk'     # 与乘客交谈
        }
    
    def build_model(self, trainable_base=False, dropout_rate=0.5):
        """构建MobileNetV2模型"""
        print("正在构建MobileNetV2模型...")
        
        # 加载预训练的MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # 设置基础模型是否可训练
        base_model.trainable = trainable_base
        
        # 构建完整模型
        inputs = tf.keras.Input(shape=(*self.img_size, 3))
        
        # 数据预处理层
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # 基础模型
        x = base_model(x, training=False)
        
        # 全局平均池化
        x = GlobalAveragePooling2D()(x)
        
        # 全连接层
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate * 0.6)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate * 0.4)(x)
        
        # 输出层
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # 创建模型
        self.model = Model(inputs, outputs)
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("模型构建完成！")
        self.model.summary()
        
        return self.model
    
    def create_callbacks(self, model_save_path, patience=10):
        """创建训练回调函数"""
        callbacks = [
            # 模型检查点
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # 早停
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # 学习率调度
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_dataset, val_dataset, epochs=50, callbacks=None):
        """训练模型"""
        print("开始训练模型...")
        
        # 训练模型
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("模型训练完成！")
        return self.history
    
    def evaluate(self, test_dataset):
        """评估模型"""
        print("评估模型性能...")
        
        # 在测试集上评估
        test_loss, test_accuracy = self.model.evaluate(test_dataset, verbose=1)
        
        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"测试集损失: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def predict(self, image):
        """预测单张图像"""
        # 预处理图像
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # 预测
        predictions = self.model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions, axis=1)[0]
        
        return predicted_class, confidence, predictions[0]
    
    def predict_batch(self, images):
        """批量预测"""
        predictions = self.model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        return predicted_classes, confidences, predictions
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        if self.history is None:
            print("没有训练历史数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.history.history['loss'], label='训练损失')
        axes[0, 0].plot(self.history.history['val_loss'], label='验证损失')
        axes[0, 0].set_title('模型损失')
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.history.history['accuracy'], label='训练准确率')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='验证准确率')
        axes[0, 1].set_title('模型准确率')
        axes[0, 1].set_xlabel('轮次')
        axes[0, 1].set_ylabel('准确率')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'])
            axes[1, 0].set_title('学习率变化')
            axes[1, 0].set_xlabel('轮次')
            axes[1, 0].set_ylabel('学习率')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # 训练进度
        epochs = range(1, len(self.history.history['loss']) + 1)
        axes[1, 1].plot(epochs, self.history.history['loss'], 'b-', label='训练损失')
        axes[1, 1].plot(epochs, self.history.history['val_loss'], 'r-', label='验证损失')
        axes[1, 1].set_title('训练进度')
        axes[1, 1].set_xlabel('轮次')
        axes[1, 1].set_ylabel('损失')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")
        
        plt.show()
    
    def save_model(self, model_path):
        """保存模型"""
        self.model.save(model_path)
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path):
        """加载模型"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"模型已从 {model_path} 加载")
        return self.model
    
    def get_class_name(self, class_id):
        """获取类别名称"""
        return self.class_names.get(class_id, f"未知类别_{class_id}")
    
    def get_all_class_names(self):
        """获取所有类别名称"""
        return [self.class_names[i] for i in range(self.num_classes)]

def create_model_summary(model):
    """创建模型摘要"""
    print("=" * 60)
    print("模型架构摘要")
    print("=" * 60)
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {non_trainable_params:,}")
    print("=" * 60)

def main():
    """测试模型"""
    # 创建模型
    model = ExtendedDriverMonitor(num_classes=10)
    model.build_model()
    
    # 创建模型摘要
    create_model_summary(model.model)
    
    return model

if __name__ == "__main__":
    main() 