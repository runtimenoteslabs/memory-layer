"""Tests for resilience utilities (retry, circuit breaker, fallback)."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_layer.core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    RetryConfig,
    calculate_backoff,
    get_circuit_breaker,
    reset_all_circuit_breakers,
    retry,
    retry_sync,
    try_multiple,
    with_fallback,
    with_fallback_sync,
    with_timeout,
)


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True


class TestCalculateBackoff:
    """Tests for calculate_backoff function."""

    def test_exponential_growth(self):
        """Test that delay grows exponentially."""
        config = RetryConfig(base_delay=1.0, jitter=False)

        delay0 = calculate_backoff(0, config)
        delay1 = calculate_backoff(1, config)
        delay2 = calculate_backoff(2, config)

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=5.0, jitter=False)

        delay = calculate_backoff(10, config)  # Would be 1024 without cap
        assert delay == 5.0

    def test_jitter_adds_variance(self):
        """Test that jitter adds variance to delay."""
        config = RetryConfig(base_delay=1.0, jitter=True)

        delays = [calculate_backoff(1, config) for _ in range(10)]

        # With jitter, not all delays should be exactly 2.0
        unique_delays = set(delays)
        assert len(unique_delays) > 1


class TestRetryDecorator:
    """Tests for retry decorator."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test successful call doesn't retry."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on transient failure."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Transient failure")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self):
        """Test raises after max attempts."""

        @retry(max_attempts=3, base_delay=0.01)
        async def always_fails():
            raise TimeoutError("Always fails")

        with pytest.raises(TimeoutError, match="Always fails"):
            await always_fails()

    @pytest.mark.asyncio
    async def test_non_recoverable_not_retried(self):
        """Test non-recoverable errors are not retried."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        async def non_recoverable():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not recoverable")

        with pytest.raises(ValueError):
            await non_recoverable()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_specific_exceptions(self):
        """Test retry only on specific exceptions."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01, exceptions=(ValueError,))
        async def specific_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry this")
            return "success"

        result = await specific_retry()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        callback_calls = []

        def on_retry(exc, attempt):
            callback_calls.append((str(exc), attempt))

        @retry(max_attempts=3, base_delay=0.01, on_retry=on_retry)
        async def flaky_func():
            if len(callback_calls) < 1:
                raise TimeoutError("Fail once")
            return "success"

        await flaky_func()
        assert len(callback_calls) == 1
        assert callback_calls[0][1] == 1


class TestRetrySyncDecorator:
    """Tests for retry_sync decorator."""

    def test_success_no_retry(self):
        """Test successful call doesn't retry."""
        call_count = 0

        @retry_sync(max_attempts=3, base_delay=0.01)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self):
        """Test retry on transient failure."""
        call_count = 0

        @retry_sync(max_attempts=3, base_delay=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Transient failure")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 2


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_closed(self):
        """Test circuit starts in closed state."""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(name="test", config=config)

        for _ in range(3):
            breaker.record_failure(Exception("fail"))

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_closed is False

    def test_success_resets_failure_count(self):
        """Test success resets failure count."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(name="test", config=config)

        breaker.record_failure(Exception("fail"))
        breaker.record_failure(Exception("fail"))
        breaker.record_success()
        breaker.record_failure(Exception("fail"))
        breaker.record_failure(Exception("fail"))

        # Should still be closed (count was reset)
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_after_timeout(self):
        """Test circuit enters half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        breaker = CircuitBreaker(name="test", config=config)

        breaker.record_failure(Exception("fail"))
        breaker.record_failure(Exception("fail"))
        assert breaker.state == CircuitState.OPEN

        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

    def test_closes_after_success_in_half_open(self):
        """Test circuit closes after successes in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=2, success_threshold=2, timeout=0.01
        )
        breaker = CircuitBreaker(name="test", config=config)

        # Open the circuit
        breaker.record_failure(Exception("fail"))
        breaker.record_failure(Exception("fail"))

        # Wait for half-open
        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN

        # Record successes
        breaker.record_success()
        breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in half-open."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.01)
        breaker = CircuitBreaker(name="test", config=config)

        # Open the circuit
        breaker.record_failure(Exception("fail"))
        breaker.record_failure(Exception("fail"))

        # Wait for half-open
        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN

        # Fail again
        breaker.record_failure(Exception("fail"))

        assert breaker.state == CircuitState.OPEN

    def test_excluded_exceptions_not_counted(self):
        """Test excluded exceptions don't count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=2, excluded_exceptions=(ValueError,)
        )
        breaker = CircuitBreaker(name="test", config=config)

        # These shouldn't count
        breaker.record_failure(ValueError("excluded"))
        breaker.record_failure(ValueError("excluded"))
        breaker.record_failure(ValueError("excluded"))

        assert breaker.state == CircuitState.CLOSED

    def test_reset(self):
        """Test manual reset."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(name="test", config=config)

        breaker.record_failure(Exception("fail"))
        breaker.record_failure(Exception("fail"))
        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Test circuit breaker decorator with success."""
        breaker = CircuitBreaker(name="test")

        @breaker
        async def successful_call():
            return "success"

        result = await successful_call()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_rejects_when_open(self):
        """Test circuit breaker decorator rejects when open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(name="test", config=config)

        @breaker
        async def failing_call():
            raise Exception("fail")

        with pytest.raises(Exception):
            await failing_call()

        # Circuit should now be open
        with pytest.raises(CircuitOpenError):
            await failing_call()


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry."""

    def test_get_circuit_breaker_creates_new(self):
        """Test get_circuit_breaker creates new breaker."""
        reset_all_circuit_breakers()
        breaker = get_circuit_breaker("unique_test_name")
        assert breaker.name == "unique_test_name"

    def test_get_circuit_breaker_returns_same(self):
        """Test get_circuit_breaker returns same instance."""
        reset_all_circuit_breakers()
        breaker1 = get_circuit_breaker("shared_name")
        breaker2 = get_circuit_breaker("shared_name")
        assert breaker1 is breaker2

    def test_reset_all_circuit_breakers(self):
        """Test reset_all_circuit_breakers."""
        reset_all_circuit_breakers()
        breaker = get_circuit_breaker(
            "reset_test",
            CircuitBreakerConfig(failure_threshold=1),
        )
        breaker.record_failure(Exception("fail"))
        assert breaker.state == CircuitState.OPEN

        reset_all_circuit_breakers()
        assert breaker.state == CircuitState.CLOSED


class TestWithFallback:
    """Tests for with_fallback decorator."""

    @pytest.mark.asyncio
    async def test_returns_result_on_success(self):
        """Test returns actual result on success."""

        @with_fallback(fallback_value="fallback")
        async def successful():
            return "real_value"

        result = await successful()
        assert result == "real_value"

    @pytest.mark.asyncio
    async def test_returns_fallback_on_failure(self):
        """Test returns fallback on failure."""

        @with_fallback(fallback_value="fallback")
        async def failing():
            raise Exception("fail")

        result = await failing()
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_only_catches_specified_exceptions(self):
        """Test only catches specified exception types."""

        @with_fallback(fallback_value="fallback", exceptions=(ValueError,))
        async def failing():
            raise TypeError("not caught")

        with pytest.raises(TypeError):
            await failing()


class TestWithFallbackSync:
    """Tests for with_fallback_sync decorator."""

    def test_returns_result_on_success(self):
        """Test returns actual result on success."""

        @with_fallback_sync(fallback_value="fallback")
        def successful():
            return "real_value"

        result = successful()
        assert result == "real_value"

    def test_returns_fallback_on_failure(self):
        """Test returns fallback on failure."""

        @with_fallback_sync(fallback_value=[])
        def failing():
            raise Exception("fail")

        result = failing()
        assert result == []


class TestTryMultiple:
    """Tests for try_multiple function."""

    @pytest.mark.asyncio
    async def test_returns_first_success(self):
        """Test returns first successful result."""

        async def op1():
            raise Exception("fail")

        async def op2():
            return "from_op2"

        async def op3():
            return "from_op3"

        result = await try_multiple(
            (op1, "op1"),
            (op2, "op2"),
            (op3, "op3"),
        )
        assert result == "from_op2"

    @pytest.mark.asyncio
    async def test_returns_default_when_all_fail(self):
        """Test returns default when all operations fail."""

        async def failing():
            raise Exception("fail")

        result = await try_multiple(
            (failing, "op1"),
            (failing, "op2"),
            default="default_value",
        )
        assert result == "default_value"

    @pytest.mark.asyncio
    async def test_returns_none_default(self):
        """Test returns None as default."""

        async def failing():
            raise Exception("fail")

        result = await try_multiple((failing, "op1"))
        assert result is None


class TestWithTimeout:
    """Tests for with_timeout function."""

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        """Test successful completion within timeout."""

        async def quick():
            await asyncio.sleep(0.01)
            return "done"

        result = await with_timeout(quick(), timeout=1.0)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self):
        """Test raises TimeoutError when exceeded."""

        async def slow():
            await asyncio.sleep(10.0)
            return "never"

        with pytest.raises(asyncio.TimeoutError):
            await with_timeout(slow(), timeout=0.01)

    @pytest.mark.asyncio
    async def test_custom_timeout_message(self):
        """Test custom timeout message."""

        async def slow():
            await asyncio.sleep(10.0)

        with pytest.raises(asyncio.TimeoutError, match="Custom message"):
            await with_timeout(slow(), timeout=0.01, timeout_message="Custom message")


class TestCircuitOpenError:
    """Tests for CircuitOpenError."""

    def test_error_message(self):
        """Test error message format."""
        error = CircuitOpenError("embedding_api")
        assert "embedding_api" in str(error)
        assert "temporarily unavailable" in str(error).lower()

    def test_is_recoverable(self):
        """Test CircuitOpenError is recoverable."""
        error = CircuitOpenError("test")
        assert error.recoverable is True
