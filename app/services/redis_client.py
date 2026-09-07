import redis
from app.core.config import settings
_redis_client: redis.Redis | None = None
def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
        )
    return _redis_client
def ping_redis() -> bool:
    try:
        client = get_redis_client()
        return client.ping()
    except redis.exceptions.RedisError:
        return False