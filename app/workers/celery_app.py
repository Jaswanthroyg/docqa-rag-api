from celery import Celery

from app.core.config import settings

celery_app = Celery("rag", broker=settings.redis_url, include=["app.workers.tasks"])
celery_app.conf.update(
    task_acks_late=True,  # a crashed worker's task is redelivered, not lost
    worker_prefetch_multiplier=1,
    task_ignore_result=True,
    task_time_limit=600,
    broker_connection_retry_on_startup=True,
)
