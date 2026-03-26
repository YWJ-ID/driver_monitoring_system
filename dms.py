import argparse
import cv2
import numpy as np
import torch
import tensorflow as tf
from ultralytics import YOLO

from dms_utils.dms_utils import load_and_preprocess_image, ACTIONS
from net import MobileNet
from facial_tracking.facialTracking import FacialTracker
import facial_tracking.conf as conf

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


def infer_one_frame(image, model, yolo_model, facial_tracker):
    eyes_status = ''
    yawn_status = ''
    actions = []  # 改为列表，支持多个行为

    facial_tracker.process_frame(image)
    if facial_tracker.detected:
        eyes_status = facial_tracker.eyes_status
        yawn_status = facial_tracker.yawn_status
        smoking_gesture = facial_tracker.smoking_gesture

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # YOLOv8推理，只检测手机(67)和香烟(74)
    yolo_result = yolo_model(rgb_image, classes=[67, 74], conf=0.25)
    
    # 优化YOLOv8检测参数，提高香烟检测准确性
    threshold = 0.25  # 降低置信度阈值，从0.45改为0.25
    
    # YOLOv8 结果处理
    phone_detected = False
    cigarette_detected = False
    cigarette_boxes = []  # 存储香烟检测框
        
    if len(yolo_result) > 0:
        boxes = yolo_result[0].boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls.item())
                confidence = box.conf.item()
                    
                if class_id == 67:  # 手机
                    phone_detected = True
                    # 绘制手机检测框
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(image, f'Phone: {confidence:.2f}', (int(x1), int(y1)-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                elif class_id == 74:  # 香烟
                    cigarette_detected = True
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cigarette_boxes.append([x1, y1, x2, y2, confidence, class_id])
                    # 绘制香烟检测框
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(image, f'Cigarette: {confidence:.2f}', (int(x1), int(y1)-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    # 检测吸烟的嘴部特征（方案B）
    smoking_mouth_detected = False
    if facial_tracker.detected and facial_tracker.lips:
        # 嘴部微张但不是打哈欠（吸烟时的特征）
        mouth_ratio = facial_tracker.lips.mouth_open_ratio
        smoking_mouth_detected = 0.1 < mouth_ratio < 0.3

    rgb_image = cv2.resize(rgb_image, (224,224))
    rgb_image = tf.expand_dims(rgb_image, 0)
    y = model.predict(rgb_image)
    result = np.argmax(y, axis=1)

    # 检测各种行为并添加到actions列表
    if result[0] == 0 and phone_detected:
        actions.append(list(ACTIONS.keys())[result[0]])
    if result[0] == 1 and phone_detected and eyes_status == 'eye closed':
        actions.append(list(ACTIONS.keys())[result[0]])
    
    # 吸烟检测逻辑（方案B + 手部检测）
    smoking_detected = False
    if cigarette_detected:
        smoking_detected = True
    elif smoking_mouth_detected and smoking_gesture == 'smoking gesture':
        smoking_detected = True
    
    if smoking_detected:
        actions.append(list(ACTIONS.keys())[2])  # 添加吸烟行为

    cv2.putText(image, f'Driver eyes: {eyes_status}', (30,40), 0, 1,conf.LM_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(image, f'Driver action: {", ".join(actions)}', (30,80), 0, 1, conf.CT_COLOR, 2, lineType=cv2.LINE_AA)
    
    return image


def infer(args):
    image_path = args.image
    video_path = args.video
    cam_id = args.webcam
    checkpoint = args.checkpoint
    save = args.save

    model = MobileNet()
    model.load_weights(checkpoint)

    yolo_model = YOLO('yolov8s')
    
    # YOLOv8配置优化，提高小物体检测准确性
    # YOLOv8会自动处理类别过滤，我们只需要设置置信度阈值
    # 注意：YOLOv8的classes参数在推理时设置，而不是模型加载时

    facial_tracker = FacialTracker()

    if image_path:
        image = cv2.imread(image_path)
        image = infer_one_frame(image, model, yolo_model, facial_tracker)
        cv2.imwrite('images/test_inferred.jpg', image)
    
    if video_path or cam_id is not None:
        cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

        if cam_id is not None:
            # 配置摄像头参数 - 使用更兼容的方式
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 降低分辨率提高稳定性
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 检查摄像头是否成功打开
            if not cap.isOpened():
                print(f"Error: Could not open camera {cam_id}")
                return
            
            # 等待摄像头预热
            import time
            time.sleep(1.0)
            
            # 尝试读取几帧来初始化摄像头
            for i in range(10):
                ret, frame = cap.read()
                if ret:
                    print(f"Camera successfully initialized after {i+1} frames")
                    break
                time.sleep(0.1)
            else:
                print("Warning: Failed to read initial frames from camera")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if save:
            out = cv2.VideoWriter('videos/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'),
                fps, (frame_width,frame_height))

        while True:
            success, image = cap.read()
            if not success:
                break

            image = infer_one_frame(image, model, yolo_model, facial_tracker)
            if save:
                out.write(image)

            cv2.imshow('DMS', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        if save:
            out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=str, default=None, help='Image path')
    p.add_argument('--video', type=str, default=None, help='Video path')
    p.add_argument('--webcam', type=int, default=None, help='Cam ID')
    p.add_argument('--checkpoint', type=str, help='Pre-trained model file path')
    p.add_argument('--save', type=bool, default=False, help='Save video or not')
    args = p.parse_args()

    infer(args)
