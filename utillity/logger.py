# logger.py
import logging

class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        self.logger = logging.getLogger("log")
        self.logger.setLevel(logging.DEBUG)
        
        # ファイルハンドラを追加
        file_handler = logging.FileHandler('app.log')  # 出力先のファイル名
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)  # ファイルハンドラを追加
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        handler.setLevel(logging.INFO)  # ターミナルにはinfoレベル以下のみを表示
        self.logger.setLevel(logging.DEBUG)  # ファイルにはデバッグ以上を記録

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            try:
                self.logger.debug('start - {} - {}'.format(func.__module__, func.__name__))
                return func(*args, **kwargs)

            except Exception as e:
                self.logger.debug('An error occurred - {} - {}'.format(func.__module__, func.__name__))
                raise e

            finally:
                self.logger.debug('end - {} - {}'.format(func.__module__, func.__name__))
                pass
        return wrapper

logger = Logger()
