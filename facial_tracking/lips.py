import cv2
import time
import facial_tracking.conf as conf

from facial_tracking.faceMesh import FaceMesh
from facial_tracking.eye import Eye


class Lips:
    """
    The object of lips, computing its features from face landmarks.

    Args:
        frame (numpy,ndarray): the input frame
        face_landmarks (mediapipe face landmarks object): contains the face landmarks coordinates
        id (list of int): the indices of lips in the landmarks
    """

    def __init__(self, frame, face_landmarks, id):

        self.frame = frame
        self.face_landmarks = face_landmarks
        self.id = id

        self.pos = self._get_lips_pos()
        self.mouth_open_ratio = self._get_open_ratio()
    
    def _get_lips_pos(self):
        """Get the positions of lips."""
        h, w = self.frame.shape[:2]
        lips_pos = list()
        for id in self.id:
            pos = self.face_landmarks.landmark[id]
            cx = int(pos.x * w)
            cy = int(pos.y * h)
            lips_pos.append([cx, cy])

        return lips_pos
    
    def _get_open_ratio(self):
        """Get the ratio of mouth open."""
        return (self.pos[3][1] - self.pos[2][1]) / (self.pos[0][0] - self.pos[1][0])
    
    def mouth_open(self, threshold=conf.MOUTH_OPEN):
        """Check whether mouth is open."""
        return self.mouth_open_ratio > threshold
    
    def detect_hand_near_mouth(self, hand_positions):
        """
        检测手是否靠近嘴部
        Args:
            hand_positions: 手部关键点位置列表 [(x1,y1), (x2,y2), ...]
        Returns:
            bool: 手是否靠近嘴部
        """
        if not hand_positions or len(self.pos) < 4:
            return False
        
        # 计算嘴部中心位置
        mouth_center_x = (self.pos[0][0] + self.pos[1][0]) // 2
        mouth_center_y = (self.pos[2][1] + self.pos[3][1]) // 2
        
        # 检查手部位置是否在嘴部附近
        for hand_pos in hand_positions:
            distance = ((hand_pos[0] - mouth_center_x) ** 2 + 
                       (hand_pos[1] - mouth_center_y) ** 2) ** 0.5
            if distance < conf.HAND_NEAR_MOUTH_DISTANCE:
                return True
        return False
    
    def detect_smoking_finger_gesture(self, finger_positions):
        """
        检测手指吸烟手势
        Args:
            finger_positions: 手指关键点位置列表，特别是食指和拇指
        Returns:
            bool: 是否检测到吸烟手指手势
        """
        if not finger_positions or len(self.pos) < 4:
            return False
        
        # 计算嘴部中心位置
        mouth_center_x = (self.pos[0][0] + self.pos[1][0]) // 2
        mouth_center_y = (self.pos[2][1] + self.pos[3][1]) // 2
        
        # 检查是否有手指靠近嘴部（吸烟手势）
        for finger_pos in finger_positions:
            distance = ((finger_pos[0] - mouth_center_x) ** 2 + 
                       (finger_pos[1] - mouth_center_y) ** 2) ** 0.5
            if distance < conf.FINGER_SMOKING_DISTANCE:
                return True
        return False
    
    def draw_lips(self):
        """Draw the target landmarks of lips."""
        for pos in self.pos:
            cv2.circle(self.frame, pos, 1, conf.LM_COLOR, -1, lineType=cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(conf.CAM_ID)
    cap.set(3, conf.FRAME_W)
    cap.set(4, conf.FRAME_H)
    fm = FaceMesh()
    ptime = 0
    ctime = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        text = ''
        
        fm.process_frame(frame)
        fm.draw_mesh_lips()
        fm.draw_mesh_eyes()
        if fm.mesh_result.multi_face_landmarks:
            for face_landmarks in fm.mesh_result.multi_face_landmarks:
                leftEye  = Eye(frame, face_landmarks, conf.LEFT_EYE)
                rightEye = Eye(frame, face_landmarks, conf.RIGHT_EYE)
                leftEye.iris.draw_iris(True)
                rightEye.iris.draw_iris(True)
                lips = Lips(frame, face_landmarks, conf.LIPS)

                if lips.mouth_open():
                    text = 'Yawning'

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {int(fps)}', (30,30), 0, 0.8,
                    conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f'{text}', (30,70), 0, 0.8,
                    conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Lips tracking', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
