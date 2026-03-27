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

# 全局变量，用于跟踪手机检测的持续时间
phone_detection_start_time = None  # 首次检测到手机的时间
PHONE_DETECTION_THRESHOLD = 2.0  # 需要持续检测的秒数

# 疲劳驾驶检测配置
FATIGUE_CONFIG = {
    'eye_closed_duration': 3.0,      # 闭眼持续超过 3 秒判定为疲劳
    'yawn_frequency': 3,             # 1 分钟内打哈欠次数 >= 3 次判定为疲劳
    'yawn_duration': 2.0,            # 单次哈欠持续时间（秒）
}

# 疲劳状态跟踪
fatigue_tracker = {
    'eye_close_start': None,         # 开始闭眼时间
    'yawn_count': 0,                 # 哈欠计数
    'last_yawn_time': None,          # 上次哈欠时间
    'is_yawning': False,             # 是否正在打哈欠
    'yawn_start_time': None,         # 哈欠开始时间
}

def infer_one_frame(image, model, yolo_model, facial_tracker, frame_timestamp=None):
    global phone_detection_start_time, PHONE_DETECTION_THRESHOLD
    global fatigue_tracker, FATIGUE_CONFIG
    
    eyes_status = ''
    yawn_status = ''
    actions = []  # 改为列表，支持多个行为

    facial_tracker.process_frame(image)
    if facial_tracker.detected:
        eyes_status = facial_tracker.eyes_status
        yawn_status = facial_tracker.yawn_status
        smoking_gesture = facial_tracker.smoking_gesture
        
        # 疲劳驾驶检测 - 基于闭眼持续时间
        current_time = frame_timestamp if frame_timestamp else time.time()
        if eyes_status == 'eye closed':
            if fatigue_tracker['eye_close_start'] is None:
                fatigue_tracker['eye_close_start'] = current_time
            else:
                closed_duration = current_time - fatigue_tracker['eye_close_start']
                if closed_duration >= FATIGUE_CONFIG['eye_closed_duration']:
                    actions.append('fatigue')  # 添加疲劳行为
                    # 在画面上显示疲劳警告 - 高科技风格
                    cv2.putText(image, f'WARNING: FATIGUE DETECTED', (30, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                    cv2.putText(image, f'Duration: {closed_duration:.1f}s', (30, 190),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        else:
            # 眼睛睁开，重置计时器
            if fatigue_tracker['eye_close_start'] is not None:
                duration = current_time - fatigue_tracker['eye_close_start']
                if duration < FATIGUE_CONFIG['eye_closed_duration']:
                    pass  # 正常眨眼，不判定为疲劳
                fatigue_tracker['eye_close_start'] = None
        
        # 打哈欠检测 - 基于嘴部开合度和频率
        if yawn_status == 'yawning':
            if not fatigue_tracker['is_yawning']:
                # 刚开始打哈欠
                fatigue_tracker['is_yawning'] = True
                fatigue_tracker['yawn_start_time'] = current_time
                # 记录哈欠次数（如果距离上次哈欠超过 5 秒，算作新的哈欠）
                if fatigue_tracker['last_yawn_time'] is None or \
                   (current_time - fatigue_tracker['last_yawn_time']) > 5.0:
                    fatigue_tracker['yawn_count'] += 1
                    fatigue_tracker['last_yawn_time'] = current_time
                    print(f"🥱 Yawn detected! Count: {fatigue_tracker['yawn_count']}")
            else:
                # 正在打哈欠中
                yawn_duration = current_time - fatigue_tracker['yawn_start_time']
                cv2.putText(image, f'YAWNING DETECTED', (30, 230),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.putText(image, f'Duration: {yawn_duration:.1f}s', (30, 270),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        else:
            # 没有打哈欠
            fatigue_tracker['is_yawning'] = False
        
        # 检查哈欠频率是否达到疲劳标准
        if fatigue_tracker['yawn_count'] >= FATIGUE_CONFIG['yawn_frequency']:
            if 'fatigue' not in actions:
                actions.append('fatigue')
            cv2.putText(image, f'ALERT: YAWN COUNT = {fatigue_tracker["yawn_count"]}', (30, 310),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3, lineType=cv2.LINE_AA)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # YOLOv8 推理，只检测手机 (67) 和香烟 (74)
    yolo_result = yolo_model(rgb_image, classes=[67, 74], conf=0.30)
        
    # 优化 YOLOv8 检测参数，提高香烟检测准确性
    threshold = 0.30  # 降低置信度阈值，从 0.45 改为 0.25
        
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
    
    # 手机检测时间窗口机制 - 只有持续检测到手机超过 2 秒才判定为玩手机
    current_time = frame_timestamp if frame_timestamp else time.time()
        
    if phone_detected:
        # 如果之前没有检测到手机，记录开始时间
        if phone_detection_start_time is None:
            phone_detection_start_time = current_time
            print(f"\n⏱️  Start detecting phone at {current_time:.2f}")
        # 检查是否持续检测到手机超过阈值
        elif current_time - phone_detection_start_time >= PHONE_DETECTION_THRESHOLD:
            # 已经持续检测到手机超过 2 秒，可以判定为玩手机
            pass  # 保持 phone_detected = True
        # 如果持续时间不足 2 秒，暂时不判定为玩手机（但继续显示检测框）
        else:
            elapsed = current_time - phone_detection_start_time
            # 在画面上显示倒计时提示 - 高科技风格
            cv2.putText(image, f'DETECTING PHONE... {elapsed:.1f}s', 
                       (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            # 暂时不判定为玩手机（用于后续的行为识别）
            phone_for_action = False
    else:
        # 没有检测到手机，重置计时器
        if phone_detection_start_time is not None:
            duration = current_time - phone_detection_start_time
            print(f"❌ Phone lost after {duration:.2f}s (< {PHONE_DETECTION_THRESHOLD}s required)")
            phone_detection_start_time = None
        phone_for_action = False
        
    # 如果已经持续检测到手机超过 2 秒，设置标志用于行为识别
    phone_for_action = (phone_detected and 
                       (phone_detection_start_time is not None and 
                        current_time - phone_detection_start_time >= PHONE_DETECTION_THRESHOLD))

    # 检测吸烟的嘴部特征（方案 B）
    smoking_mouth_detected = False
    if facial_tracker.detected and facial_tracker.lips:
        # 嘴部微张但不是打哈欠（吸烟时的特征）
        mouth_ratio = facial_tracker.lips.mouth_open_ratio
        smoking_mouth_detected = 0.1 < mouth_ratio < 0.3
    
    rgb_image = cv2.resize(rgb_image, (224,224))
    rgb_image = tf.expand_dims(rgb_image, 0)
    y = model.predict(rgb_image)
    result = np.argmax(y, axis=1)
    
    # 检测各种行为并添加到 actions 列表
    # 使用 phone_for_action 替代 phone_detected，确保只有持续 2 秒才判定
    if result[0] == 0 and phone_for_action:
        actions.append(list(ACTIONS.keys())[result[0]])
    if result[0] == 1 and phone_for_action and eyes_status == 'eye closed':
        actions.append(list(ACTIONS.keys())[result[0]])
    
    # 吸烟检测逻辑（方案B + 手部检测）
    smoking_detected = False
    if cigarette_detected:
        smoking_detected = True
    elif smoking_mouth_detected and smoking_gesture == 'smoking gesture':
        smoking_detected = True
    
    if smoking_detected:
        actions.append(list(ACTIONS.keys())[2])  # 添加吸烟行为

    cv2.putText(image, f'EYES STATUS: {eyes_status.upper()}', (30,40), 0, 1.2, (0, 255, 0), 3, lineType=cv2.LINE_AA)
    action_text = ', '.join(actions).upper() if actions else 'SAFE DRIVING'
    action_color = (0, 0, 255) if actions else (0, 255, 0)
    cv2.putText(image, f'ACTION: {action_text}', (30,80), 0, 1.2, action_color, 3, lineType=cv2.LINE_AA)
    
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
        
        # 调试信息：检查实际获取的帧率
        print(f"Camera info - Resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}")
        
        # 如果 FPS 为 0 或异常，使用默认值 30
        if fps <= 0 or fps > 60:
            print(f"Warning: Invalid FPS value ({fps}), using default 30.0")
            fps = 30.0

        if save:
            # 关键修改：使用动态帧率，根据实际 AI 处理能力调整
            # 这样保存的视频播放速度才正常
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # 初始使用摄像头 FPS（后面会根据实际帧率调整）
            # 使用时间戳生成唯一文件名，避免覆盖
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f'videos/output_{timestamp}.avi'
            out = cv2.VideoWriter(output_filename, fourcc, 
                                  fps, (frame_width, frame_height))
            print(f"Video Writer initialized at {fps:.1f} FPS")
            print(f"Output file: {output_filename}")
            print("Note: Video will be saved with actual capture rate for correct playback speed")

        # 用于统计帧数和性能
        frame_count = 0
        saved_frame_count = 0
        import time
        start_time = time.time()

        while True:
            success, image = cap.read()
            if not success:
                break

            # 获取当前帧的时间戳，用于手机检测计时
            frame_timestamp = time.time()
            
            # 传递时间戳到推理函数
            image = infer_one_frame(image, model, yolo_model, facial_tracker, frame_timestamp)
            
            # 先显示画面并检查退出（保证能及时响应）
            cv2.imshow('DMS', image)
            # 增加等待时间到 50ms，确保键盘事件能被捕获
            # 在 AI 推理慢的情况下，需要更长的等待时间
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                print("\n⚠️  Exit requested by user...")
                break
            
            # 然后再保存（不影响退出响应）
            if save:
                out.write(image)
                saved_frame_count += 1
            
            frame_count += 1
        
        # 打印统计信息
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # 先释放资源，确保文件可以被访问
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
