# ============================================================================
# 文件: core/scheduler.py
# ============================================================================
"""
后台调度器 - 使用schedule库实现自动任务调度

功能:
- 交易日10:30和15:15自动调用incremental_update_with_realtime()
- 周末18:00执行完整检查
- 支持start/stop方法
- 调度日志记录
"""
import time
import logging
import threading
from datetime import datetime
from typing import Optional
import schedule
from .updater import DataUpdater
from config import settings


class DataScheduler:
    """
    后台数据调度器
    
    功能:
    - 交易日定时更新（10:30, 15:15）
    - 周末完整检查（18:00）
    - 后台运行
    - 调度日志记录
    
    使用:
    ```python
    scheduler = DataScheduler()
    scheduler.start()
    
    # 或者指定自定义时间
    scheduler = DataScheduler()
    scheduler.add_custom_job('daily_check', 'weekly', 'sunday', '18:00')
    scheduler.start()
    ```
    """
    
    def __init__(self):
        """初始化调度器"""
        self.updater = DataUpdater()
        self.logger = logging.getLogger("DataScheduler")
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        
        # 调度器状态
        self.stats = {
            'jobs_scheduled': 0,
            'jobs_executed': 0,
            'jobs_failed': 0,
            'last_execution': None,
            'last_error': None
        }
    
    def setup_default_jobs(self):
        """设置默认调度任务"""
        # 清空现有任务
        schedule.clear()
        
        # 交易日10:30更新
        schedule.every().monday.at(settings.scheduler.UPDATE_TIME_MORNING).do(self._job_incremental_update).tag('daily_update')
        schedule.every().tuesday.at(settings.scheduler.UPDATE_TIME_MORNING).do(self._job_incremental_update).tag('daily_update')
        schedule.every().wednesday.at(settings.scheduler.UPDATE_TIME_MORNING).do(self._job_incremental_update).tag('daily_update')
        schedule.every().thursday.at(settings.scheduler.UPDATE_TIME_MORNING).do(self._job_incremental_update).tag('daily_update')
        schedule.every().friday.at(settings.scheduler.UPDATE_TIME_MORNING).do(self._job_incremental_update).tag('daily_update')
        
        # 交易日15:15更新
        schedule.every().monday.at(settings.scheduler.UPDATE_TIME_AFTERNOON).do(self._job_incremental_update).tag('daily_update')
        schedule.every().tuesday.at(settings.scheduler.UPDATE_TIME_AFTERNOON).do(self._job_incremental_update).tag('daily_update')
        schedule.every().wednesday.at(settings.scheduler.UPDATE_TIME_AFTERNOON).do(self._job_incremental_update).tag('daily_update')
        schedule.every().thursday.at(settings.scheduler.UPDATE_TIME_AFTERNOON).do(self._job_incremental_update).tag('daily_update')
        schedule.every().friday.at(settings.scheduler.UPDATE_TIME_AFTERNOON).do(self._job_incremental_update).tag('daily_update')
        
        # 周末完整检查（周日18:00）
        schedule.every().sunday.at(settings.scheduler.WEEKLY_CHECK_TIME).do(self._job_weekly_check).tag('weekly_check')
        
        self.stats['jobs_scheduled'] = len(schedule.jobs)
        self.logger.info(f"Scheduled {self.stats['jobs_scheduled']} jobs: {schedule.jobs}")
    
    def _job_incremental_update(self):
        """增量更新任务"""
        self.logger.info("=" * 60)
        self.logger.info("SCHEDULED: Starting incremental update with realtime...")
        
        try:
            result = self.updater.incremental_update_with_realtime()
            self.stats['jobs_executed'] += 1
            self.stats['last_execution'] = datetime.now()
            
            self.logger.info(f"SCHEDULED: Incremental update completed: {result}")
            
            # 更新成功日志
            self._log_execution('incremental_update', result, success=True)
            
        except Exception as e:
            self.stats['jobs_failed'] += 1
            self.stats['last_error'] = str(e)
            self.logger.error(f"SCHEDULED: Incremental update failed: {e}")
            
            # 错误日志
            self._log_execution('incremental_update', {'error': str(e)}, success=False)
        
        self.logger.info("=" * 60)
    
    def _job_weekly_check(self):
        """周度完整检查任务"""
        self.logger.info("=" * 60)
        self.logger.info("SCHEDULED: Starting weekly full check...")
        
        try:
            # 执行完整性检查
            integrity_result = self.updater.check_integrity()
            self.logger.info(f"SCHEDULED: Integrity check: {integrity_result}")
            
            # 如果有缺失的股票，执行补全
            if integrity_result.get('incomplete_stocks', 0) > 0:
                self.logger.info("SCHEDULED: Found incomplete stocks, running full update...")
                full_result = self.updater.full_update()
                self.logger.info(f"SCHEDULED: Full update completed: {full_result}")
            else:
                full_result = {'mode': 'weekly_check', 'message': 'All stocks complete'}
            
            self.stats['jobs_executed'] += 1
            self.stats['last_execution'] = datetime.now()
            
            self.logger.info(f"SCHEDULED: Weekly check completed: {full_result}")
            
            # 更新成功日志
            self._log_execution('weekly_check', {
                'integrity': integrity_result,
                'full_update': full_result
            }, success=True)
            
        except Exception as e:
            self.stats['jobs_failed'] += 1
            self.stats['last_error'] = str(e)
            self.logger.error(f"SCHEDULED: Weekly check failed: {e}")
            
            # 错误日志
            self._log_execution('weekly_check', {'error': str(e)}, success=False)
        
        self.logger.info("=" * 60)
    
    def _log_execution(self, job_type: str, result: dict, success: bool):
        """记录执行日志"""
        log_entry = {
            'timestamp': datetime.now(),
            'job_type': job_type,
            'success': success,
            'result': result,
            'stats': self.stats.copy()
        }
        
        # 写入日志文件
        self._write_execution_log(log_entry)
        
        # 也输出到控制台
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"[{job_type.upper()}] {status}: {result}")
    
    def _write_execution_log(self, log_entry: dict):
        """写入执行日志到文件"""
        try:
            import json
            from pathlib import Path
            
            log_file = settings.path.LOG_DIR / "scheduler_execution.log"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'timestamp': log_entry['timestamp'].isoformat(),
                    'job_type': log_entry['job_type'],
                    'success': log_entry['success'],
                    'result': log_entry['result'],
                    'stats': log_entry['stats']
                }, ensure_ascii=False, default=str))
                f.write('\n')
                
        except Exception as e:
            self.logger.warning(f"Failed to write execution log: {e}")
    
    def add_custom_job(self, job_name: str, frequency: str, day: str = None, time_str: str = None):
        """
        添加自定义任务
        
        Args:
            job_name: 任务名称
            frequency: 频率 ('daily', 'weekly')
            day: 星期几 ('monday', 'tuesday', ...)，仅当frequency='weekly'时需要
            time_str: 时间字符串 ('HH:MM')
        """
        if frequency == 'daily':
            if not time_str:
                raise ValueError("time_str is required for daily jobs")
            getattr(schedule.every(), day or 'day').at(time_str).do(self._custom_job, job_name).tag(f'custom_{job_name}')
        
        elif frequency == 'weekly':
            if not day or not time_str:
                raise ValueError("day and time_str are required for weekly jobs")
            getattr(schedule.every(), day).at(time_str).do(self._custom_job, job_name).tag(f'custom_{job_name}')
        
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        self.logger.info(f"Added custom job: {job_name} ({frequency}, {day or 'daily'}, {time_str})")
    
    def _custom_job(self, job_name: str):
        """自定义任务"""
        self.logger.info(f"SCHEDULED: Custom job '{job_name}' started")
        
        try:
            # 这里可以添加自定义逻辑
            result = {'job_name': job_name, 'status': 'custom_job_executed'}
            
            self.stats['jobs_executed'] += 1
            self.stats['last_execution'] = datetime.now()
            
            self.logger.info(f"SCHEDULED: Custom job '{job_name}' completed: {result}")
            
        except Exception as e:
            self.stats['jobs_failed'] += 1
            self.stats['last_error'] = str(e)
            self.logger.error(f"SCHEDULED: Custom job '{job_name}' failed: {e}")
    
    def start(self):
        """启动调度器"""
        if self._running:
            self.logger.warning("Scheduler is already running")
            return
        
        # 设置默认任务
        if not settings.scheduler.ENABLED:
            self.logger.info("Scheduler is disabled in settings")
            return
        
        self.setup_default_jobs()
        
        # 启动后台线程
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        
        self.logger.info("DataScheduler started successfully")
    
    def stop(self):
        """停止调度器"""
        if not self._running:
            self.logger.warning("Scheduler is not running")
            return
        
        self._running = False
        schedule.clear()
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        self.logger.info("DataScheduler stopped")
    
    def _run_scheduler(self):
        """运行调度器"""
        self.logger.info("Scheduler thread started")
        
        while self._running:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
        
        self.logger.info("Scheduler thread stopped")
    
    def get_status(self) -> dict:
        """获取调度器状态"""
        return {
            'running': self._running,
            'next_runs': [str(job.next_run) for job in schedule.jobs],
            'stats': self.stats,
            'jobs_count': len(schedule.jobs)
        }
    
    def get_upcoming_jobs(self) -> list:
        """获取即将运行的任务"""
        upcoming = []
        for job in schedule.jobs:
            upcoming.append({
                'job': str(job.job_func),
                'next_run': job.next_run,
                'tags': list(job.tags)
            })
        return upcoming
    
    def pause_job(self, job_tag: str):
        """暂停指定标签的任务"""
        schedule.pause_job(job_tag)
        self.logger.info(f"Paused jobs with tag: {job_tag}")
    
    def resume_job(self, job_tag: str):
        """恢复指定标签的任务"""
        schedule.resume_job(job_tag)
        self.logger.info(f"Resumed jobs with tag: {job_tag}")
    
    def clear_jobs(self):
        """清空所有任务"""
        schedule.clear()
        self.logger.info("All scheduled jobs cleared")


# ==================== 快捷接口 ====================
def start_scheduler():
    """启动调度器"""
    scheduler = DataScheduler()
    scheduler.start()
    return scheduler


def stop_scheduler():
    """停止调度器"""
    # 这里需要全局调度器实例
    # 实际使用中建议使用单例模式
    pass


# ==================== 单元测试 ====================
if __name__ == "__main__":
    import unittest
    
    class TestDataScheduler(unittest.TestCase):
        """数据调度器测试"""
        
        def test_scheduler_setup(self):
            """测试调度器设置"""
            scheduler = DataScheduler()
            scheduler.setup_default_jobs()
            
            # 检查是否有任务被安排
            self.assertGreater(len(schedule.jobs), 0)
            
            # 检查统计信息
            self.assertGreaterEqual(scheduler.stats['jobs_scheduled'], 0)
        
        def test_custom_job(self):
            """测试自定义任务"""
            scheduler = DataScheduler()
            scheduler.add_custom_job('test_job', 'daily', time_str='12:00')
            
            # 检查任务是否被添加
            self.assertEqual(len(schedule.jobs), 1)
        
        def test_status(self):
            """测试状态获取"""
            scheduler = DataScheduler()
            status = scheduler.get_status()
            
            self.assertIn('running', status)
            self.assertIn('stats', status)
            self.assertIn('jobs_count', status)
    
    # 运行测试
    print("Running DataScheduler Tests...")
    unittest.main(verbosity=2, exit=False)