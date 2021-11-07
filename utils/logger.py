##################
# Import modules #
##################

import logging
import os

#######################
# Classes & Functions #
#######################

def get_logger(file_path : str):
    """
        Logger를 Return 해주는 함수 입니다.
    """
    # log file이 없다면 생성
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("*--------Create Log File--------*")

    logger = logging.getLogger(__name__)

    # 로그의 출력 기준 설정
    logger.setLevel(logging.INFO)

    # log 출력 형식
    formatter = logging.Formatter('%(asctime)s: %(message)s')

    # log 출력
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # log를 파일에 출력
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger