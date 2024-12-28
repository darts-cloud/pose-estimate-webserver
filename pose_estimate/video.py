import threading
import queue
import numpy as np
import cv2
import time
import datetime
# from playbackCamera.CountFps import *
from utillity.logger import *
import os

"""
映像リソースより最新のフレームを読み込むためのクラス
FPSが異なる複数リソースにアクセスする場合、
常に最新のフレームを取得することができる。
"""
class VideoCapture():

	"""コンストラクタ"""
	def __init__(self, src, dest=None, fps=None, width=None, height=None):
		try:
			self.src = int(src)
		except ValueError:
			self.src = str(src)
		logger.info(f"Connect: Camera:{str(self.src)}")

		self.video = cv2.VideoCapture(self.src)

		self.fps = fps
		self.width = width
		self.height = height
		if self.fps is None:
			self.fps = self.video.get(cv2.CAP_PROP_FPS)
		if self.width is None:
			self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
		if self.height is None:
			self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

		logger.info(f"Video Param: FPS={int(self.fps)}, Width={int(self.width)}, Height={int(self.height)}")

		if not self.video.isOpened():
			logger.error(f"Connect Error.")
			return

		print("Connected.")
		self.stopped = False
		self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		if dest is not None:
			self.output_setting(dest)
		
	def output_setting(self, output_path):

		path = output_path
		if os.path.isdir(path):
			path = os.path.join(path, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
			path = path + "_" + ".mp4"
			# path = path + "_" + os.path.splitext(os.path.basename(self.src))[0] + ".mp4"
		logger.info(f"output path: {path}")

		fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
		self.out = cv2.VideoWriter(path, fourcc, self.fps,
                          (int(self.width), int(self.height)))  # 出力先のファイルを開く

	"""
	Queueより最新のフレームを取得する。
	"""
	def read(self):
		return self.video.read()
	
	def imshow(self, frame):
		cv2.imshow("Tracking", frame)  # フレームを画面表示
		if self.out is not None:
			self.out.write(frame)
	
	def write(self, frame):
		if self.out is not None:
			self.out.write(frame)

	"""映像出力を止める。"""
	def stop(self):
		self.stopped = True
	
	"""映像リソースをリリースする。"""
	def release(self):
		self.stopped = True
		self.video.release()
		self.out.release()
		cv2.destroyAllWindows()
	
	"""映像リソースが開けているか、返す。"""
	def isOpened(self):
		return self.video.isOpened()

	"""OpenCvで設定できる映像リソース情報を設定。"""
	def set(self, va1, va2):
		self.video.set(va1, va2)

	"""OpenCvで取得できる映像リソース情報を返す。"""
	def get(self, i):
		return self.video.get(i)

"""
映像リソースより最新のフレームを読み込むためのクラス
FPSが異なる複数リソースにアクセスする場合、
常に最新のフレームを取得することができる。
"""
class ThreadingVideoCapture(VideoCapture):

	"""コンストラクタ"""
	def __init__(self, src, max_queue_size=256):
		super().__init__(src)

		self.q = queue.Queue(maxsize=max_queue_size)
		self.stopped = False
#		self.fpsCount = CountFps()
		thread = threading.Thread(target=self.update, daemon=True)
		thread.start()

		self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		self.bef = None


	def output_setting(self, output_path):
		width = self.get(cv2.CAP_PROP_FRAME_WIDTH)
		height = self.get(cv2.CAP_PROP_FRAME_HEIGHT)
		self.set(cv2.CAP_PROP_FPS, 30) # フレームレート(fps)
		frame_rate = int(self.get(cv2.CAP_PROP_FPS)) # フレームレート(fps)
		print(frame_rate)

		fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
		self.out = cv2.VideoWriter(output_path, fourcc, frame_rate,
                          (int(width), int(height)))  # 出力先のファイルを開く

	"""
	Queueより最新のフレームを取得する。
	"""
	def read(self):
		if not self.q.empty():
			try:
				# 常に最新のフレームを読み込む
				ret = self.q.get_nowait()
				self.bef = ret
				return True, ret
			except queue.Empty:
				return False, None

		if self.bef is None:
			ret = self.q.get()
			self.bef = ret
			return True, ret

		return True, self.bef


	""" 
	ソース元より、常に動画を受け取り、
	最新の動画のみ、Queueに登録する。
	別スレッドで実行する。
	"""
	def update(self):
		
		while True:

			try:
				if self.stopped:
					time.sleep(10)
					self.video = cv2.VideoCapture(self.src)
					if self.video.isOpened():
						print("ReConnect.")
						self.stopped = False
					continue
				
				ret, img = self.video.read()
#				self.fpsCount.CountFrame()

				if not ret:
					self.stop()
					print(self.src + ":stop")
					continue
				
				"""
				OpenCvは内部バッファーを持っている。
				常に最新のフレームを取得したい場合、
				このバッファーが邪魔となるため、
				常に全フレームを読み込み、不要となるフレームを
				内部的に読み込むことで内部バッファー内の映像を
				全て吐き出している。
				"""
				if not self.q.empty():
					try:
						# 常に最新のフレームを読み込む
						self.q.get_nowait()
					except queue.Empty:
						pass

				times = time.time()
#				fps = self.fpsCount.CountFps()
				fps = 0
				self.q.put(img)
			except Exception:
				pass
	
# class VideoPlayer():
    
#     def __init__(self):
#         super().__init__()
#         # self.setWindowTitle('Video Player')
#         # self.layout = QVBoxLayout()
#         # self.label = QLabel()
#         # self.layout.addWidget(self.label)
#         # self.setLayout(self.layout)

#     def show_frame(self, frame):
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         height, width, channel = frame.shape
#         bytesPerLine = 3 * width
#         # qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
#         # pixmap = QPixmap.fromImage(qImg)
#         # self.label.setPixmap(pixmap)