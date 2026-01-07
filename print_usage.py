import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from prettytable import PrettyTable
import pytz
import locale

def connect_to_db(db_path="model_usage.db"):
    """连接到数据库"""
    conn = sqlite3.connect(db_path)
    return conn

def get_last_n_days_data(conn, days=7):
    """获取过去n天的数据"""
    # 获取当前UTC时间
    now = datetime.now(pytz.UTC)
    # 计算n天前的UTC时间
    n_days_ago = now - timedelta(days=days)
    
    query = """
    SELECT 
        timestamp, 
        model_id, 
        input_tokens, 
        output_tokens, 
        total_fee,
        COALESCE(user_id, 'NULL') as user_id
    FROM model_usage
    WHERE timestamp >= ?
    """
    
    # 将查询结果加载到Pandas DataFrame
    df = pd.read_sql_query(query, conn, params=(n_days_ago.strftime('%Y-%m-%d %H:%M:%S'),))
    
    # 将timestamp转换为datetime类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 将UTC时间转换为UTC+8时间
    df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.UTC)
    df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Shanghai')
    
    # 创建日期列，只保留日期部分
    df['date'] = df['timestamp'].dt.date
    
    # 将total_fee除以1,000,000
    df['total_fee'] = df['total_fee'] / 1_000_000
    
    return df

def daily_user_fee_stats(df):
    """统计过去7天，每天每个user的total fee"""
    # 按日期和用户分组，计算total_fee总和
    daily_user_fees = df.groupby(['date', 'user_id'])['total_fee'].sum().reset_index()
    
    # 计算每个用户的总费用以便排序
    user_total_fees = daily_user_fees.groupby('user_id')['total_fee'].sum().sort_values(ascending=False)
    
    # 按总费用排序的用户列表
    sorted_users = user_total_fees.index.tolist()
    
    # 按日期和用户分组，计算total_fee总和，并按用户总费用排序列
    pivot = pd.pivot_table(
        data=df,
        values='total_fee',
        index='date',
        columns='user_id',
        aggfunc='sum',
        fill_value=0
    )
    
    # 重新排序列，按用户总费用降序
    pivot = pivot[sorted_users]
    
    # 添加日合计列
    pivot['总计'] = pivot.sum(axis=1)
    
    # 添加用户合计行
    pivot.loc['总计'] = pivot.sum()
    
    return pivot

def model_token_fee_stats(df):
    """统计过去7天，每个模型的总计input token, output token和total fee"""
    # 按模型分组，计算input_tokens, output_tokens和total_fee的总和
    result = df.groupby('model_id').agg({
        'input_tokens': 'sum',
        'output_tokens': 'sum',
        'total_fee': 'sum'
    }).reset_index()
    
    # 按total_fee降序排序
    result = result.sort_values('total_fee', ascending=False)
    
    return result

def top_models_daily_fee(df, top_n=3):
    """统计total fee前N名的模型，每天每个模型的total fee"""
    # 获取total fee前N名的模型和它们的总费用
    top_models_series = df.groupby('model_id')['total_fee'].sum().nlargest(top_n)
    top_models = top_models_series.index.tolist()
    
    # 筛选这些模型的数据
    top_models_df = df[df['model_id'].isin(top_models)]
    
    # 创建透视表
    pivot = pd.pivot_table(
        data=top_models_df,
        values='total_fee',
        index='date',
        columns='model_id',
        aggfunc='sum',
        fill_value=0
    )
    
    # 按模型总费用排序列
    pivot = pivot[top_models]
    
    # 添加日合计列
    pivot['总计'] = pivot.sum(axis=1)
    
    # 添加模型合计行
    pivot.loc['总计'] = pivot.sum()
    
    return pivot

def format_currency(value):
    """将数值格式化为货币形式"""
    return f"¥{value:.2f}"

def format_number(value):
    """将数值格式化为带千分位的形式"""
    return f"{value:,}"

def dataframe_to_prettytable(df, title=None, is_money_table=True):
    """将DataFrame转换为PrettyTable格式"""
    pt = PrettyTable()
    
    # 如果有标题，设置标题
    if title:
        pt.title = title
    
    # 设置字段名
    if isinstance(df.index, pd.MultiIndex):
        # 处理多级索引
        pt.field_names = [' | '.join(map(str, df.index.names))] + list(df.columns)
    else:
        pt.field_names = [df.index.name if df.index.name else '索引'] + list(df.columns)
    
    # 添加行
    for row_idx, row in df.iterrows():
        if isinstance(row_idx, tuple):
            # 多级索引
            row_values = [' | '.join(map(str, row_idx))]
        else:
            row_values = [row_idx]
            
        # 格式化值
        for col in df.columns:
            value = row[col]
            if col == 'total_fee' or (is_money_table and col != 'input_tokens' and col != 'output_tokens'):
                # 金额格式
                row_values.append(format_currency(value))
            elif col in ['input_tokens', 'output_tokens']:
                # 千分位格式
                row_values.append(format_number(value))
            else:
                row_values.append(value)
                
        pt.add_row(row_values)
    
    # 设置对齐方式
    for i, field in enumerate(pt.field_names):
        if i == 0:  # 索引列左对齐
            pt.align[field] = 'l'
        elif field in ['input_tokens', 'output_tokens', 'total_fee'] or field == '总计' or is_money_table:
            # 数字列右对齐
            pt.align[field] = 'r'
        else:
            pt.align[field] = 'l'
    
    return pt


def main():
    """主函数"""
    # 连接数据库
    conn = connect_to_db()
    
    days = 14
    # 获取过去{days}天的数据
    df = get_last_n_days_data(conn, days=days)
    
    # 1. 统计过去{days}天，每天每个user的total fee
    user_fee_pivot = daily_user_fee_stats(df)
    user_fee_table = dataframe_to_prettytable(user_fee_pivot, f"过去{days}天每天每个用户的消费统计(单位:CNY)")
    print(user_fee_table)
    print("\n")
    
    # 2. 统计过去{days}天，每个模型的总计input token, output token和total fee
    model_stats = model_token_fee_stats(df)
    model_stats_table = dataframe_to_prettytable(model_stats, f"过去{days}天各模型使用统计", is_money_table=False)
    print(model_stats_table)
    print("\n")
    
    # 3. total fee前三名的模型，统计每天每个模型的total fee
    top_models_pivot = top_models_daily_fee(df)
    top_models_table = dataframe_to_prettytable(top_models_pivot, f"过去{days}天Top3模型每日消费统计(单位:CNY)")
    print(top_models_table)
    
    # 关闭数据库连接
    conn.close()

if __name__ == "__main__":
    main()