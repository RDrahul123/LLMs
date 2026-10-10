"""
Rate Limiter & Cost Control — Production-grade rate limiting and circuit breaker.

Implements:
  - Token bucket rate limiter
  - Sliding window rate limiter
  - Per-user rate limiting
  - Token budget tracking (daily/monthly limits)
  - Cost calculator (tokens × model pricing)
  - Retry with exponential backoff
  - Circuit breaker pattern
  - Decorators for easy use

Run: python rate_limiter.py
"""

import asyncio
import functools
import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Pricing (USD per 1M tokens)
# ---------------------------------------------------------------------------

MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


# ---------------------------------------------------------------------------
# Cost Calculator
# ---------------------------------------------------------------------------


class CostCalculator:
    """Calculate costs for LLM API calls."""

    @staticmethod
    def calculate(
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 8)

    @staticmethod
    def format_cost(cost_usd: float) -> str:
        if cost_usd < 0.01:
            return f"${cost_usd * 1000:.2f}m"
        return f"${cost_usd:.4f}"


# ---------------------------------------------------------------------------
# Token Bucket Rate Limiter
# ---------------------------------------------------------------------------


class TokenBucketRateLimiter:
    """
    Token bucket algorithm: tokens are added at a fixed rate up to a maximum.
    Each request consumes tokens. If no tokens available, request is rejected.
    """

    def __init__(
        self,
        max_tokens: int = 60,
        refill_rate: float = 1.0,  # tokens per second
    ):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = max_tokens
        self.last_refill = time.time()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def acquire(self, tokens: int = 1) -> bool:
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_time(self, tokens: int = 1) -> float:
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.refill_rate


# ---------------------------------------------------------------------------
# Sliding Window Rate Limiter
# ---------------------------------------------------------------------------


class SlidingWindowRateLimiter:
    """
    Sliding window: counts requests in a rolling time window.
    More accurate than fixed windows for burst detection.
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: float = 60.0,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: list[float] = []

    def _cleanup(self):
        cutoff = time.time() - self.window_seconds
        self.requests = [t for t in self.requests if t > cutoff]

    def allow(self) -> bool:
        self._cleanup()
        if len(self.requests) < self.max_requests:
            self.requests.append(time.time())
            return True
        return False

    def remaining(self) -> int:
        self._cleanup()
        return max(0, self.max_requests - len(self.requests))

    def retry_after(self) -> float:
        self._cleanup()
        if len(self.requests) < self.max_requests:
            return 0.0
        oldest = self.requests[0]
        return max(0, oldest + self.window_seconds - time.time())


# ---------------------------------------------------------------------------
# Per-User Rate Limiter
# ---------------------------------------------------------------------------


class PerUserRateLimiter:
    """Rate limiter that tracks limits per user."""

    def __init__(
        self,
        max_requests: int = 30,
        window_seconds: float = 60.0,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, list[float]] = defaultdict(list)

    def _cleanup(self, user_id: str):
        cutoff = time.time() - self.window_seconds
        self.user_windows[user_id] = [
            t for t in self.user_windows[user_id] if t > cutoff
        ]

    def allow(self, user_id: str) -> bool:
        self._cleanup(user_id)
        if len(self.user_windows[user_id]) < self.max_requests:
            self.user_windows[user_id].append(time.time())
            return True
        return False

    def remaining(self, user_id: str) -> int:
        self._cleanup(user_id)
        return max(0, self.max_requests - len(self.user_windows[user_id]))


# ---------------------------------------------------------------------------
# Token Budget Tracker
# ---------------------------------------------------------------------------


@dataclass
class UsageRecord:
    tokens: int
    cost_usd: float
    timestamp: float


class TokenBudget:
    """Track token usage and enforce daily/monthly budgets."""

    def __init__(
        self,
        daily_token_limit: int = 1_000_000,
        monthly_token_limit: int = 10_000_000,
        daily_cost_limit: float = 10.0,
        monthly_cost_limit: float = 100.0,
    ):
        self.daily_token_limit = daily_token_limit
        self.monthly_token_limit = monthly_token_limit
        self.daily_cost_limit = daily_cost_limit
        self.monthly_cost_limit = monthly_cost_limit
        self.records: dict[str, list[UsageRecord]] = defaultdict(list)

    def _day_key(self) -> str:
        return time.strftime("%Y-%m-%d")

    def _month_key(self) -> str:
        return time.strftime("%Y-%m")

    def _cleanup(self, user_id: str):
        cutoff_month = time.time() - 30 * 86400
        self.records[user_id] = [
            r for r in self.records[user_id] if r.timestamp > cutoff_month
        ]

    def can_use(self, user_id: str, estimated_tokens: int, model: str) -> bool:
        self._cleanup(user_id)
        records = self.records[user_id]

        # Daily checks
        day_start = time.time() - 86400
        daily_records = [r for r in records if r.timestamp > day_start]
        daily_tokens = sum(r.tokens for r in daily_records)
        daily_cost = sum(r.cost_usd for r in daily_records)

        if daily_tokens + estimated_tokens > self.daily_token_limit:
            logger.warning(f"User {user_id} would exceed daily token limit")
            return False
        if daily_cost + CostCalculator.calculate(model, estimated_tokens, 0) > self.daily_cost_limit:
            logger.warning(f"User {user_id} would exceed daily cost limit")
            return False

        # Monthly checks
        month_start = time.time() - 30 * 86400
        month_records = [r for r in records if r.timestamp > month_start]
        month_tokens = sum(r.tokens for r in month_records)
        month_cost = sum(r.cost_usd for r in month_records)

        if month_tokens + estimated_tokens > self.monthly_token_limit:
            return False
        if month_cost + CostCalculator.calculate(model, estimated_tokens, 0) > self.monthly_cost_limit:
            return False

        return True

    def record_usage(
        self,
        user_id: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ):
        total = input_tokens + output_tokens
        cost = CostCalculator.calculate(model, input_tokens, output_tokens)
        self.records[user_id].append(
            UsageRecord(tokens=total, cost_usd=cost, timestamp=time.time())
        )
        logger.info(
            f"Usage: user={user_id} model={model} "
            f"tokens={total} cost={CostCalculator.format_cost(cost)}"
        )

    def get_usage(self, user_id: str) -> dict:
        self._cleanup(user_id)
        records = self.records[user_id]

        day_start = time.time() - 86400
        month_start = time.time() - 30 * 86400

        daily = [r for r in records if r.timestamp > day_start]
        monthly = [r for r in records if r.timestamp > month_start]

        return {
            "daily_tokens": sum(r.tokens for r in daily),
            "daily_cost_usd": sum(r.cost_usd for r in daily),
            "monthly_tokens": sum(r.tokens for r in monthly),
            "monthly_cost_usd": sum(r.cost_usd for r in monthly),
            "daily_token_limit": self.daily_token_limit,
            "monthly_token_limit": self.monthly_token_limit,
            "daily_cost_limit": self.daily_cost_limit,
            "monthly_cost_limit": self.monthly_cost_limit,
        }


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Prevents cascading failures by stopping requests when failures exceed threshold.

    States:
      CLOSED  → Normal. Requests pass through.
      OPEN    → Too many failures. All requests rejected.
      HALF_OPEN → Recovery test. One request allowed through.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_count = 0

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_count = 0
                logger.info("Circuit breaker: OPEN -> HALF_OPEN")
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_count < self.half_open_max
        return False

    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
        else:
            self.failure_count = 0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.info("Circuit breaker: HALF_OPEN -> OPEN")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker: CLOSED -> OPEN "
                f"(failures={self.failure_count})"
            )

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
        }


# ---------------------------------------------------------------------------
# Retry with Exponential Backoff
# ---------------------------------------------------------------------------


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> Any:
    """
    Retry a function with exponential backoff.

    Args:
        func: Async callable to retry.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap.
        exponential_base: Base for exponential growth.
        jitter: Add random jitter to prevent thundering herd.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                break

            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            if jitter:
                delay += random.uniform(0, delay * 0.5)

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                f"Retrying in {delay:.2f}s"
            )
            await asyncio.sleep(delay)

    raise last_exception


def retry_sync(
    func: Callable,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> Any:
    """Synchronous version of retry_with_backoff."""
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                break

            delay = min(base_delay * (2 ** attempt), max_delay)
            delay += random.uniform(0, delay * 0.5)
            time.sleep(delay)

    raise last_exception


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def rate_limit(
    max_requests: int = 30,
    window_seconds: float = 60.0,
):
    """Decorator to apply per-user rate limiting."""
    limiter = SlidingWindowRateLimiter(max_requests, window_seconds)

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(user_id: str = "default", *args, **kwargs):
            if not limiter.allow():
                retry_after = limiter.retry_after()
                raise RateLimitExceeded(
                    f"Rate limit exceeded. Retry after {retry_after:.1f}s"
                )
            return await func(user_id=user_id, *args, **kwargs)
        return wrapper
    return decorator


def circuit_protected(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
):
    """Decorator to apply circuit breaker pattern."""
    breaker = CircuitBreaker(failure_threshold, recovery_timeout)

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not breaker.allow_request():
                raise CircuitOpenError("Circuit breaker is open")
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        return wrapper
    return decorator


def budget_check(
    daily_token_limit: int = 1_000_000,
    monthly_token_limit: int = 10_000_000,
):
    """Decorator to enforce token budgets."""
    budget = TokenBudget(daily_token_limit, monthly_token_limit)

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(user_id: str = "default", estimated_tokens: int = 1000, *args, **kwargs):
            if not budget.can_use(user_id, estimated_tokens, "gpt-4o"):
                raise BudgetExceeded(f"Token budget exceeded for user {user_id}")
            return await func(user_id=user_id, estimated_tokens=estimated_tokens, *args, **kwargs)
        return wrapper
    return budget


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class RateLimitExceeded(Exception):
    pass


class CircuitOpenError(Exception):
    pass


class BudgetExceeded(Exception):
    pass


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo_token_bucket():
    print("\n=== Token Bucket Rate Limiter ===")
    limiter = TokenBucketRateLimiter(max_tokens=5, refill_rate=1.0)

    for i in range(8):
        allowed = limiter.acquire()
        print(f"Request {i + 1}: {'Allowed' if allowed else 'Denied'} "
              f"(tokens remaining: {limiter.tokens:.1f})")
        if i >= 4:
            time.sleep(0.3)


def demo_sliding_window():
    print("\n=== Sliding Window Rate Limiter ===")
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=2.0)

    for i in range(8):
        allowed = limiter.allow()
        print(f"Request {i + 1}: {'Allowed' if allowed else 'Denied'} "
              f"(remaining: {limiter.remaining()})")
        if i == 4:
            print("  Waiting 2.5s for window to reset...")
            time.sleep(2.5)


def demo_circuit_breaker():
    print("\n=== Circuit Breaker ===")
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)

    def failing_function():
        raise ConnectionError("Simulated failure")

    for i in range(10):
        allowed = breaker.allow_request()
        if allowed:
            try:
                failing_function()
            except ConnectionError:
                breaker.record_failure()
        print(f"Request {i + 1}: state={breaker.state.value}, "
              f"failures={breaker.failure_count}")
        if i == 6:
            print("  Waiting 2.5s for recovery...")
            time.sleep(2.5)


def demo_cost_calculator():
    print("\n=== Cost Calculator ===")
    cases = [
        ("gpt-4o", 1000, 500),
        ("gpt-4o-mini", 1000, 500),
        ("gpt-4-turbo", 1000, 500),
    ]
    for model, inp, out in cases:
        cost = CostCalculator.calculate(model, inp, out)
        print(f"  {model}: {inp} in + {out} out = {CostCalculator.format_cost(cost)}")


def demo_budget():
    print("\n=== Token Budget ===")
    budget = TokenBudget(
        daily_token_limit=5000,
        monthly_token_limit=50000,
        daily_cost_limit=0.10,
        monthly_cost_limit=1.0,
    )
    user = "test_user"

    for i in range(5):
        can = budget.can_use(user, 1200, "gpt-4o")
        if can:
            budget.record_usage(user, 1000, 200, "gpt-4o")
            print(f"  Request {i + 1}: Allowed")
        else:
            print(f"  Request {i + 1}: Denied (budget exceeded)")

    usage = budget.get_usage(user)
    print(f"  Usage: {usage['daily_tokens']} tokens, "
          f"{CostCalculator.format_cost(usage['daily_cost_usd'])}")


if __name__ == "__main__":
    demo_cost_calculator()
    demo_token_bucket()
    demo_sliding_window()
    demo_circuit_breaker()
    demo_budget()
    print("\nAll demos complete.")
