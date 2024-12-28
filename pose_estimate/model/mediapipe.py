# FrameProcessor.py
import cv2
from enum import *
import mediapipe as mp
from abc import ABC, abstractmethod
from pose_estimate.adapter.pose import *

class MediapipePoseModel():

    def __init__(self):
        # MediaPipe Poseモデルの初期化
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def pose_estimation(self, frame):
        # BGR画像をRGBに変換
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # パフォーマンス向上のために画像の書き込みを停止
        frame.flags.writeable = False
        # ポーズ推定を実行
        results = self.pose.process(frame)

        # ポーズ推定結果を描画
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())


        return frame

class MediapipePoseAndHandsModel():

    def __init__(self):
        super().__init__()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)

    def pose_estimation(self, frame):
        frame, _ = super().pose_estimation(frame)
        # BGR画像をRGBに変換
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # パフォーマンス向上のために画像の書き込みを停止
        frame.flags.writeable = False
        # Hands検出の実行
        results_hands = self.hands.process(frame)

        # ポーズ推定結果を描画
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # if results_hands.multi_hand_landmarks:
        #     for hand_landmarks in results_hands.multi_hand_landmarks:
        #         self.mp_drawing.draw_landmarks(
        #             frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        if results_hands.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results_hands.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)

        return frame

class MediapipePoseModel2:

    def __init__(self, size):
        super().__init__()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
                            min_detection_confidence=0.8,
                            min_tracking_confidence=0.8,
                            static_image_mode=False,
                            model_complexity=1)
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = None
        self.size = size

    def pose_estimation(self, frame):
        # BGR画像をRGBに変換
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # パフォーマンス向上のために画像の書き込みを停止
        frame.flags.writeable = False
        # Hands検出の実行
        results = self.holistic.process(frame)

        # ポーズ推定結果を描画
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # ポーズ、手、顔のランドマークの描画
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.right_hand_landmarks,
                connections=self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        # if results.left_hand_landmarks:
        #     self.mp_drawing.draw_landmarks(
        #         image=frame,
        #         landmark_list=results.left_hand_landmarks,
        #         connections=self.mp_holistic.HAND_CONNECTIONS,
        #         landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        # self.mp_drawing.draw_landmarks(
        #     frame, results.face_landmarks, self.mp_holistic.FACE_CONNECTIONS)
        # self.mp_drawing.draw_landmarks(
        #     frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        # self.mp_drawing.draw_landmarks(
        #     frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        # self.mp_drawing.draw_landmarks(
        #     frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        if results.pose_landmarks is None:
            return frame
        
        self.pose = PoseMeidapipeAdapter(self.size, results.pose_landmarks.landmark)
        if results.right_hand_landmarks:
            self.pose.right_hand = PoseMeidapipeHands(self.size, results.right_hand_landmarks.landmark)
        else:
            self.pose.right_hand = PoseMeidapipeHands(self.size, None)
        
        return frame