import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any


class ModelUsageDB:
    """管理模型使用情况的SQLite数据库"""
    
    def __init__(self, db_path: str = "model_usage.db"):
        """
        初始化数据库连接
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def open(self):
        """打开数据库连接并初始化表结构"""
        # 检查是否已经连接
        if self.conn is not None:
            return
            
        # 建立连接
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # 创建表（如果不存在）
        self._create_tables()
        
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def _create_tables(self):
        """创建必要的表结构（如果不存在）"""
        # 检查表是否已存在
        self.cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='model_usage'
        """)
        
        if not self.cursor.fetchone():
            # 创建表
            self.cursor.execute("""
                CREATE TABLE model_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_id TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    input_price INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    output_multiplier INTEGER NOT NULL,
                    total_fee INTEGER NOT NULL,
                    user_id TEXT
                )
            """)
            
            # 创建索引
            self.cursor.execute("CREATE INDEX idx_user ON model_usage (user_id)")
            self.cursor.execute("CREATE INDEX idx_model ON model_usage (model_id)")
            self.cursor.execute("CREATE INDEX idx_timestamp ON model_usage (timestamp)")
            
            self.conn.commit()
    
    def add_usage(self, 
                  model_id: str, 
                  input_tokens: int, 
                  input_price: int, 
                  output_tokens: int, 
                  output_multiplier: int, 
                  total_fee: int,
                  user_id: Optional[str] = None,
                  timestamp: Optional[datetime] = None):
        """
        添加模型使用记录
        
        Args:
            model_id: 模型标识符
            input_tokens: 输入token数量
            input_price: 输入价格
            output_multiplier: 输出乘数
            output_price: 输出价格
            user_id: 用户ID（可选）
            timestamp: 时间戳（可选，默认为当前时间）
        """
        if not self.conn:
            self.open()
            
        if timestamp is None:
            self.cursor.execute("""
                INSERT INTO model_usage 
                (model_id, input_tokens, input_price, output_tokens, output_multiplier, total_fee, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (model_id, input_tokens, input_price, output_tokens, output_multiplier, total_fee, user_id))
        else:
            # 如果提供了时间戳，则使用它
            self.cursor.execute("""
                INSERT INTO model_usage 
                (model_id, input_tokens, input_price, output_tokens, output_multiplier, total_fee, user_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (model_id, input_tokens, input_price, output_tokens, output_multiplier, total_fee, user_id, timestamp))
            
        self.conn.commit()
        
    def get_usage_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取特定用户的所有使用记录
        
        Args:
            user_id: 用户ID
        
        Returns:
            包含用户使用记录的字典列表
        """
        if not self.conn:
            self.open()
            
        self.cursor.execute("""
            SELECT * FROM model_usage WHERE user_id = ? ORDER BY timestamp DESC
        """, (user_id,))
        
        columns = [description[0] for description in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def get_usage_by_model(self, model_id: str) -> List[Dict[str, Any]]:
        """
        获取特定模型的所有使用记录
        
        Args:
            model_id: 模型ID
        
        Returns:
            包含模型使用记录的字典列表
        """
        if not self.conn:
            self.open()
            
        self.cursor.execute("""
            SELECT * FROM model_usage WHERE model_id = ? ORDER BY timestamp DESC
        """, (model_id,))
        
        columns = [description[0] for description in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def get_usage_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        获取特定日期范围内的使用记录
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            包含指定日期范围内使用记录的字典列表
        """
        if not self.conn:
            self.open()
            
        self.cursor.execute("""
            SELECT * FROM model_usage 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        """, (start_date, end_date))
        
        columns = [description[0] for description in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        
    def __enter__(self):
        """上下文管理器入口"""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()