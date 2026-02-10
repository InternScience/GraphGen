import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class RateLimiter:
    def __init__(self, qps: float = 10, burst: int = 5):
        self.qps = qps
        self.burst = burst
        self.tokens = burst
        self.last = time.time()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.time()
            self.tokens = min(
                float(self.burst), self.tokens + (now - self.last) * self.qps
            )
            self.last = now

            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.qps
                time.sleep(sleep_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class HTTPClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 30,
        qps: float = 10,
        max_concurrent: int = 5,
        headers: Optional[Dict] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.limiter = RateLimiter(qps=qps, burst=max_concurrent)

        self.session = requests.Session()
        self.session.headers.update(
            headers or {"User-Agent": "GraphGen/1.0", "Accept": "application/json"}
        )

        adapter = HTTPAdapter(
            pool_connections=max_concurrent,
            pool_maxsize=max_concurrent * 2,
            max_retries=3,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self._executor = None
        self._max_workers = max_concurrent

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def get(self, endpoint: str) -> dict:
        self.limiter.acquire()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    async def aget(self, endpoint: str) -> dict:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.get, endpoint)

    def close(self):
        self.session.close()
        if self._executor:
            self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
