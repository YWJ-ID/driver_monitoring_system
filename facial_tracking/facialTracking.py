import cv2
import time
import facial_tracking.conf as conf
import mediapipe as mp

from facial_tracking.faceMesh import FaceMesh
from facial_tracking.eye import Eye
from facial_tracking.lips import Lips


class FacialTracker:
    """
    The object of facial tracking, predicting status of eye, iris, and mouth.
    """

    def __init__(self):

        self.fm = FaceMesh()
        self.left_eye  = None
        self.right_eye = None
        self.lips = None
        self.left_eye_closed_frames  = 0
        self.right_eye_closed_frames = 0
        
        # 初始化手部检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def process_frame(self, frame):
        """Process the frame to analyze facial status."""
        self.detected = False
        self.fm.process_frame(frame)
        # self.fm.draw_mesh_lips()

        # 手部检测
        self.hand_positions = []
        self.finger_positions = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                h, w = frame.shape[:2]
                # 获取手部中心位置
                hand_center_x = int(hand_landmarks.landmark[9].x * w)  # 中指根部
                hand_center_y = int(hand_landmarks.landmark[9].y * h)
                self.hand_positions.append([hand_center_x, hand_center_y])
                
                # 获取食指和拇指位置（吸烟手势的关键点）
                index_finger_x = int(hand_landmarks.landmark[8].x * w)  # 食指尖
                index_finger_y = int(hand_landmarks.landmark[8].y * h)
                thumb_x = int(hand_landmarks.landmark[4].x * w)  # 拇指尖
                thumb_y = int(hand_landmarks.landmark[4].y * h)
                
                self.finger_positions.extend([[index_finger_x, index_finger_y], 
                                            [thumb_x, thumb_y]])

        if self.fm.mesh_result.multi_face_landmarks:
            self.detected = True
            for face_landmarks in self.fm.mesh_result.multi_face_landmarks:
                self.left_eye  = Eye(frame, face_landmarks, conf.LEFT_EYE)
                self.right_eye = Eye(frame, face_landmarks, conf.RIGHT_EYE)
                self.lips = Lips(frame, face_landmarks, conf.LIPS)
                self._check_eyes_status()
                self._check_yawn_status()
                self._check_smoking_gesture()
    
    def _check_eyes_status(self):
        self.eyes_status = ''
        
        if self.left_eye.eye_closed():
            self.left_eye_closed_frames += 1
        else:
            self.left_eye_closed_frames = 0
            # self.left_eye.iris.draw_iris(True)

        if self.right_eye.eye_closed():
            self.right_eye_closed_frames += 1
        else:
            self.right_eye_closed_frames = 0
            # self.right_eye.iris.draw_iris(True)
        
        if self._left_eye_closed() or self._right_eye_closed():
            self.eyes_status = 'eye closed'
            return
        
        if not self.left_eye.eye_closed() and not self.right_eye.eye_closed():
            if   self.left_eye.gaze_right()  and self.right_eye.gaze_right():
                self.eyes_status = 'gazing right'
            elif self.left_eye.gaze_left()   and self.right_eye.gaze_left():
                self.eyes_status = 'gazing left'
            elif self.left_eye.gaze_center() and self.right_eye.gaze_center():
                self.eyes_status = 'gazing center'

    def _check_yawn_status(self):
        self.yawn_status = ''
        if self.lips.mouth_open():
            self.yawn_status = 'yawning'
    
    def _left_eye_closed(self, threshold=conf.FRAME_CLOSED):
        return self.left_eye_closed_frames > threshold
    
    def _right_eye_closed(self, threshold=conf.FRAME_CLOSED):
        return self.right_eye_closed_frames > threshold
        
    def _check_smoking_gesture(self):
        """检测吸烟手势"""
        self.smoking_gesture = ''
        
        if self.lips and self.hand_positions:
            # 检测手是否靠近嘴部
            hand_near_mouth = self.lips.detect_hand_near_mouth(self.hand_positions)
            
            # 检测手指吸烟手势
            finger_smoking = self.lips.detect_smoking_finger_gesture(self.finger_positions)
            
            if hand_near_mouth or finger_smoking:
                self.smoking_gesture = 'smoking gesture'

def main():
    cap = cv2.VideoCapture(conf.CAM_ID)
    cap.set(3, conf.FRAME_W)
    cap.set(4, conf.FRAME_H)
    facial_tracker = FacialTracker()
    ptime = 0
    ctime = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        facial_tracker.process_frame(frame)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {int(fps)}', (30,30), 0, 0.6,
                    conf.TEXT_COLOR, 1, lineType=cv2.LINE_AA)
        
        if facial_tracker.detected:
            cv2.putText(frame, f'{facial_tracker.eyes_status}', (30,70), 0, 0.8,
                        conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f'{facial_tracker.yawn_status}', (30,110), 0, 0.8,
                        conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Facial tracking', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
