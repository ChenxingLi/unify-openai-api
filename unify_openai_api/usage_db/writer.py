import threading
import queue
from typing import Optional
from datetime import datetime
from .sql import ModelUsageDB  # 假设之前的类保存在这个模块中
import logging

logger = logging.getLogger(__name__)

class AsyncDBWriter:
    """异步数据库写入器，使用线程和队列实现MPSC模式"""
    
    def __init__(self, db_path: str = "model_usage.db"):
        """
        初始化异步写入器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self.worker_thread = None
        
    def start(self):
        """启动消费者线程"""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
            
        self.worker_thread = threading.Thread(
            target=self._db_writer_worker,
            daemon=True
        )
        self.worker_thread.start()
        
    def stop(self):
        """停止消费者线程"""
        self._stop_event.set()
        self.queue.join()  # 等待队列中所有任务完成
            
    def add_usage(self, 
                 model_id: str, 
                 input_tokens: int, 
                 input_price: float, 
                 output_tokens: int, 
                 output_price: float, 
                 user_id: Optional[str] = None,
                 timestamp: Optional[datetime] = None):
        total_fee = input_tokens * input_price + output_tokens * output_price
        
        self.queue.put(dict(
            model_id = model_id,
            input_tokens = input_tokens,
            input_price = round(input_price * 1000),
            output_tokens = output_tokens,
            output_multiplier = round(output_price / input_price * 10),
            total_fee = round(total_fee),
            user_id = user_id,
            timestamp = timestamp
        ))
        
    def _db_writer_worker(self):
        """消费者线程工作函数"""
        with ModelUsageDB(self.db_path) as db:
            while not self._stop_event.is_set() or not self.queue.empty():
                try:
                    # 设置超时以避免无限等待
                    data = self.queue.get()
                    
                    # 使用已实现的数据库方法写入
                    db.add_usage(**data)
                    self.queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error writing to database: {e}")