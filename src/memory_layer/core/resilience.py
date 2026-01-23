"""
Resilience utilities for Memory Layer.

Provides:
- Retry with exponential backoff
- Circuit breaker pattern
- Graceful degradation helpers
"""

import asyncio
import functools
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ParamSpec, TypeVar

from .exceptions import CircuitOpenError, is_recoverable

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# =============================================================================
# Retry with Exponential Backoff
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd


def calculate_backoff(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate delay for a retry attempt.

    Uses exponential backoff with optional jitter.

    Args:
        attempt: The current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds before the next retry
    """
    delay = config.base_delay * (config.exponential_base**attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add up to 25% jitter
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] | None = None,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for retrying async functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exception types to retry on. If None, retries
            on all recoverable exceptions.
        on_retry: Optional callback called before each retry with
            (exception, attempt_number)

    Example:
        @retry(max_attempts=3, base_delay=1.0)
        async def fetch_data():
            ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
    )

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    should_retry = False
                    if exceptions is not None:
                        should_retry = isinstance(e, exceptions)
                    else:
                        should_retry = is_recoverable(e)

                    if not should_retry or attempt >= config.max_attempts - 1:
                        raise

                    # Calculate backoff delay
                    delay = calculate_backoff(attempt, config)

                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed "
                        f"for {func.__name__}: {e}. Retrying in {delay:.2f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt + 1)

                    await asyncio.sleep(delay)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic error")

        return wrapper

    return decorator


def retry_sync(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] | None = None,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for retrying sync functions with exponential backoff.

    Same as retry() but for synchronous functions.
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
    )

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    should_retry = False
                    if exceptions is not None:
                        should_retry = isinstance(e, exceptions)
                    else:
                        should_retry = is_recoverable(e)

                    if not should_retry or attempt >= config.max_attempts - 1:
                        raise

                    delay = calculate_backoff(attempt, config)

                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed "
                        f"for {func.__name__}: {e}. Retrying in {delay:.2f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt + 1)

                    time.sleep(delay)

            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic error")

        return wrapper

    return decorator


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 60.0  # Seconds before trying half-open
    excluded_exceptions: tuple[type[Exception], ...] = ()  # Don't count these


@dataclass
class CircuitBreaker:
    """Circuit breaker for protecting external service calls.

    The circuit breaker prevents cascading failures by:
    1. Tracking failures and successes
    2. Opening the circuit (rejecting requests) after too many failures
    3. Periodically testing if the service has recovered
    4. Closing the circuit when service is healthy again

    Example:
        breaker = CircuitBreaker(name="embedding_api")

        @breaker
        async def call_embedding_api():
            ...
    """

    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for timeout transition."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.config.timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info(f"Circuit breaker '{self.name}' entering half-open state")
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (allowing requests)."""
        return self.state == CircuitState.CLOSED

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' closed (service recovered)")
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        # Don't count excluded exceptions
        if isinstance(exception, self.config.excluded_exceptions):
            return

        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately opens
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker '{self.name}' opened (failure in half-open state)"
            )
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' opened "
                    f"(reached {self._failure_count} failures)"
                )

    def __call__(
        self, func: Callable[P, Awaitable[T]]
    ) -> Callable[P, Awaitable[T]]:
        """Decorator to wrap an async function with circuit breaker."""

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if self.state == CircuitState.OPEN:
                raise CircuitOpenError(self.name)

            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapper

    def call_sync(self, func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to wrap a sync function with circuit breaker."""

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if self.state == CircuitState.OPEN:
                raise CircuitOpenError(self.name)

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset")


# =============================================================================
# Graceful Degradation
# =============================================================================


def with_fallback(
    fallback_value: T,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    log_level: int = logging.WARNING,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator that returns a fallback value on failure.

    Useful for graceful degradation where partial functionality
    is better than total failure.

    Args:
        fallback_value: Value to return on failure
        exceptions: Exception types to catch
        log_level: Logging level for failures

    Example:
        @with_fallback(fallback_value=[])
        async def get_optional_data():
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                logger.log(
                    log_level,
                    f"Function {func.__name__} failed, using fallback: {e}",
                )
                return fallback_value

        return wrapper

    return decorator


def with_fallback_sync(
    fallback_value: T,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    log_level: int = logging.WARNING,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Synchronous version of with_fallback."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.log(
                    log_level,
                    f"Function {func.__name__} failed, using fallback: {e}",
                )
                return fallback_value

        return wrapper

    return decorator


async def try_multiple(
    *operations: tuple[Callable[[], Awaitable[T]], str],
    default: T | None = None,
) -> T | None:
    """Try multiple operations in order, returning first success.

    Useful for fallback chains where you want to try multiple
    approaches to get data.

    Args:
        operations: Tuples of (async_callable, description)
        default: Value to return if all operations fail

    Example:
        result = await try_multiple(
            (fetch_from_cache, "cache"),
            (fetch_from_api, "api"),
            (fetch_from_backup, "backup"),
            default=None,
        )
    """
    for operation, description in operations:
        try:
            result = await operation()
            logger.debug(f"Operation '{description}' succeeded")
            return result
        except Exception as e:
            logger.debug(f"Operation '{description}' failed: {e}")
            continue

    logger.warning("All operations failed, returning default")
    return default


# =============================================================================
# Timeout Utilities
# =============================================================================


async def with_timeout(
    coro: Awaitable[T],
    timeout: float,
    timeout_message: str | None = None,
) -> T:
    """Execute a coroutine with a timeout.

    Args:
        coro: The coroutine to execute
        timeout: Timeout in seconds
        timeout_message: Custom message for timeout error

    Returns:
        The result of the coroutine

    Raises:
        asyncio.TimeoutError: If the operation times out
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        msg = timeout_message or f"Operation timed out after {timeout}s"
        raise asyncio.TimeoutError(msg) from None


# =============================================================================
# Shared Circuit Breakers (Singletons)
# =============================================================================

# Pre-configured circuit breakers for common external services
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get or create a named circuit breaker.

    Circuit breakers are cached by name, so the same breaker
    is returned for the same name across the application.

    Args:
        name: Unique name for the circuit breaker
        config: Configuration (only used on first creation)

    Returns:
        The circuit breaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name=name,
            config=config or CircuitBreakerConfig(),
        )
    return _circuit_breakers[name]


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers to closed state."""
    for breaker in _circuit_breakers.values():
        breaker.reset()
