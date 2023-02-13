import threading
import time
import os

def delete_file(file_path: str, timeout: int):
    time.sleep(timeout)
    os.remove(file_path)

def working(file_path: str):
    thread = threading.Thread(target=delete_file, args=(file_path, 10))
    thread.start()