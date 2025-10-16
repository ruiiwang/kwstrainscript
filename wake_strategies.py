import numpy as np
from collections import deque

# 基础配置
class WakeStrategy:
    """唤醒策略基类，所有策略都应继承此类"""
    def __init__(self, min_interval=1.6):
        self.min_interval = min_interval  # 最小唤醒间隔（秒）
        self.last_trigger_time = -min_interval      # 上次唤醒时间
        self.prob_history = []            # 概率历史
        self.prob_timestamps = []         # 时间戳历史
        self.name = "BaseStrategy"            # 策略名称
    
    def update_history(self, prob, timestamp):
        """更新概率历史和时间戳"""
        self.prob_history.append(prob)
        self.prob_timestamps.append(timestamp)
    
    def clean_history(self, current_time, window_s):
        """清理超出时间窗口的历史数据"""
        while (len(self.prob_timestamps) > 0 and 
               current_time - self.prob_timestamps[0] > window_s):
            self.prob_history.pop(0)
            self.prob_timestamps.pop(0)
    
    def check_trigger(self, current_time):
        """检查是否满足唤醒条件"""
        # 子类必须实现此方法
        raise NotImplementedError("子类必须实现check_trigger方法")
    
    def reset(self):
        """重置策略状态"""
        self.prob_history = []
        self.prob_timestamps = []
        self.last_trigger_time = -self.min_interval


# 策略1：连续概率策略
class ConsecutiveStrategy(WakeStrategy):
    """连续概率策略：时间窗口内所有帧概率连续超过阈值"""
    def __init__(self, consecutive_threshold=0.6, consecutive_window_ms=200, min_interval=1.6):
        super().__init__(min_interval)
        self.consecutive_threshold = consecutive_threshold
        self.consecutive_window_ms = consecutive_window_ms
        self.consecutive_window_s = consecutive_window_ms / 1000.0
        self.name = "Consecutive"
        self.required_frames = int(self.consecutive_window_ms / 20)  # Assuming 20ms hop size

    def check_trigger(self, current_time):
        """检查是否满足唤醒条件"""
        # 检查最小间隔
        if current_time - self.last_trigger_time < self.min_interval:
            return False, ""
        
        # 清理过旧的历史数据
        self.clean_history(current_time, self.consecutive_window_s)
        
        # 仅当历史记录足够长时才进行判断
        if len(self.prob_history) < self.required_frames:
            return False, ""

        # 检查窗口内是否所有帧都超过阈值
        if all(p > self.consecutive_threshold for p in self.prob_history):
            self.last_trigger_time = current_time
            return True, "Over Threshold"
        
        return False, ""


# 策略2：平均概率策略
class AverageStrategy(WakeStrategy):
    """平均概率策略：时间窗口内平均概率超过阈值"""
    def __init__(self, average_threshold=0.7, average_window_ms=240, min_interval=1.6):
        super().__init__(min_interval)
        self.average_threshold = average_threshold
        self.average_window_ms = average_window_ms
        self.average_window_s = average_window_ms / 1000.0
        self.name = "Average"
        self.required_frames = int(self.average_window_ms / 20)  # Assuming 20ms hop size
    
    def check_trigger(self, current_time):
        """检查是否满足唤醒条件"""
        # 检查最小间隔
        if current_time - self.last_trigger_time < self.min_interval:
            return False, ""
        
        # 清理过旧的历史数据
        self.clean_history(current_time, self.average_window_s)
        
        # 仅当历史记录足够长时才进行判断
        if len(self.prob_history) < self.required_frames:
            return False, ""
        
        # 计算窗口内的平均概率
        avg_prob = sum(self.prob_history) / len(self.prob_history)
        if avg_prob > self.average_threshold:
            self.last_trigger_time = current_time
            return True, "Average Probability"
        
        return False, ""


# 策略3：峰值策略
class PeakStrategy(WakeStrategy):
    """峰值策略：单帧峰值和持续时间"""
    def __init__(self, peak_threshold=0.9, duration_threshold=0.8, 
                 duration_window_ms=200, min_interval=1.6):
        super().__init__(min_interval)
        self.peak_threshold = peak_threshold
        self.duration_threshold = duration_threshold
        self.duration_window_ms = duration_window_ms
        self.duration_window_s = duration_window_ms / 1000.0
        self.name = "Peak"
        self.required_frames = int(self.duration_window_ms / 20)  # Assuming 20ms hop size
    
    def check_trigger(self, current_time):
        """检查是否满足唤醒条件"""
        # 检查最小间隔
        if current_time - self.last_trigger_time < self.min_interval:
            return False, ""
        
        # 清理过旧的历史数据
        self.clean_history(current_time, self.duration_window_s)
        
        # 仅当历史记录足够长时才进行判断
        if len(self.prob_history) < self.required_frames:
            return False, ""
        
        # 条件：任意帧概率超过峰值阈值，且窗口内平均概率超过持续阈值
        # 检查是否有峰值
        max_prob = max(self.prob_history)
        if max_prob > self.peak_threshold:
            # 检查持续时间条件
            avg_prob = sum(self.prob_history) / len(self.prob_history)
            if avg_prob > self.duration_threshold:
                self.last_trigger_time = current_time
                return True, "Peak + Duration"
        
        return False, ""


# 策略4：组合策略
class ComboStrategy(WakeStrategy):
    """组合策略：多条件组合和加权平均"""
    def __init__(self, peak_threshold=0.95, avg_threshold=0.9, 
                 duration_ms=180, weights=None, min_interval=1.6):
        super().__init__(min_interval)
        self.peak_threshold = peak_threshold
        self.avg_threshold = avg_threshold
        self.duration_ms = duration_ms
        self.duration_s = duration_ms / 1000.0
        # 默认权重：越近的样本权重越大
        self.weights = weights if weights else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.name = "Combo"
        self.required_frames = int(self.duration_ms / 20)  # Assuming 20ms hop size
    
    def check_trigger(self, current_time):
        """检查是否满足唤醒条件"""
        # 检查最小间隔
        if current_time - self.last_trigger_time < self.min_interval:
            return False, ""
        
        # 清理过旧的历史数据
        self.clean_history(current_time, self.duration_s)
        
        if len(self.prob_history) < self.required_frames:  # 需要足够的数据点
            return False, ""
        
        # 条件1：峰值超过阈值且平均值超过阈值
        combo_triggered = False
        max_prob = max(self.prob_history)
        avg_prob = sum(self.prob_history) / len(self.prob_history)
        
        if max_prob > self.peak_threshold and avg_prob > self.avg_threshold:
            combo_triggered = True
        
        # 条件2：加权平均超过阈值
        weighted_triggered = False
        # 取最近的几个样本进行加权平均
        recent_probs = self.prob_history[-len(self.weights):]
        if len(recent_probs) == len(self.weights):
            weighted_avg = sum(p * w for p, w in zip(recent_probs, self.weights)) / sum(self.weights)
            if weighted_avg > self.peak_threshold:
                weighted_triggered = True
        
        # 返回是否触发和触发类型
        if combo_triggered and weighted_triggered:
            self.last_trigger_time = current_time
            if combo_triggered and weighted_triggered:
                trigger_type = "Peak + Weighted"
            elif combo_triggered:
                trigger_type = "Peak"
            else:  # weighted_triggered
                trigger_type = "Weighted Average"
            return True, trigger_type
        
        return False, ""


# 策略工厂：用于创建和管理策略
class StrategyFactory:
    """策略工厂：创建和管理唤醒策略"""
    @staticmethod
    def create_strategy(strategy_name, **kwargs):
        """创建指定名称的策略"""
        strategies = {
            "consecutive": ConsecutiveStrategy,
            "average": AverageStrategy,
            "peak": PeakStrategy,
            "combo": ComboStrategy,
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"未知策略名称: {strategy_name}")
        
        return strategies[strategy_name](**kwargs)
    
    @staticmethod
    def get_all_strategies(min_interval=1.6):
        """获取所有策略的实例"""
        return {
            "consecutive": ConsecutiveStrategy(min_interval=min_interval),
            "average": AverageStrategy(min_interval=min_interval),
            "peak": PeakStrategy(min_interval=min_interval),
            "combo": ComboStrategy(min_interval=min_interval),
        }
