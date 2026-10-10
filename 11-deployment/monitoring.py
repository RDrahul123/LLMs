"""
Monitoring & Observability — Structured logging, metrics, and dashboards.

Implements:
  - Structured logging (JSON logs)
  - Request/response logging
  - Latency tracking (p50, p95, p99)
  - Token usage tracking
  - Cost tracking per request
  - Error rate monitoring
  - Dashboard data generation
  - Alerting thresholds
  - Prometheus-compatible metrics

Run: python monitoring.py
"""

import json
import logging
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured Logger (JSON)
# ---------------------------------------------------------------------------


class StructuredLogger:
    """JSON-formatted structured logging for production systems."""

    def __init__(self, service_name: str = "llm-service", log_file: str | None = None):
        self.service_name = service_name
        self.logger = logging.getLogger(f"structured.{service_name}")
        self.logger.setLevel(logging.INFO)
        self._lock = threading.Lock()
        self._log_buffer: list[dict] = []
        self.max_buffer = 10000

        if log_file:
            handler = logging.FileHandler(log_file)
        else:
            handler = logging.StreamHandler()

        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _emit(self, level: str, event: str, **kwargs):
        entry = {
            "timestamp": time.time(),
            "service": self.service_name,
            "level": level,
            "event": event,
            **kwargs,
        }
        with self._lock:
            self._log_buffer.append(entry)
            if len(self._log_buffer) > self.max_buffer:
                self._log_buffer = self._log_buffer[-self.max_buffer:]

        self.logger.info(json.dumps(entry, default=str))
        return entry

    def info(self, event: str, **kwargs):
        return self._emit("info", event, **kwargs)

    def warning(self, event: str, **kwargs):
        return self._emit("warning", event, **kwargs)

    def error(self, event: str, **kwargs):
        return self._emit("error", event, **kwargs)

    def debug(self, event: str, **kwargs):
        return self._emit("debug", event, **kwargs)

    def get_recent(self, n: int = 100) -> list[dict]:
        with self._lock:
            return self._log_buffer[-n:]


# ---------------------------------------------------------------------------
# Request/Response Logger
# ---------------------------------------------------------------------------


class RequestLogger:
    """Log request/response details for debugging and auditing."""

    def __init__(self, structured_logger: StructuredLogger):
        self.logger = structured_logger

    def log_request(
        self,
        request_id: str,
        endpoint: str,
        model: str,
        user_id: str | None = None,
        input_tokens: int = 0,
        **kwargs,
    ):
        return self.logger.info(
            "llm_request",
            request_id=request_id,
            endpoint=endpoint,
            model=model,
            user_id=user_id,
            input_tokens=input_tokens,
            **kwargs,
        )

    def log_response(
        self,
        request_id: str,
        model: str,
        output_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: float = 0,
        status: str = "success",
        **kwargs,
    ):
        return self.logger.info(
            "llm_response",
            request_id=request_id,
            model=model,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            status=status,
            **kwargs,
        )

    def log_error(
        self,
        request_id: str,
        error_type: str,
        error_message: str,
        model: str | None = None,
        **kwargs,
    ):
        return self.logger.error(
            "llm_error",
            request_id=request_id,
            error_type=error_type,
            error_message=error_message,
            model=model,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Latency Tracker
# ---------------------------------------------------------------------------


class LatencyTracker:
    """Track latency percentiles over sliding windows."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies: list[float] = []
        self._lock = threading.Lock()

    def record(self, latency_ms: float):
        with self._lock:
            self.latencies.append(latency_ms)
            if len(self.latencies) > self.window_size:
                self.latencies = self.latencies[-self.window_size:]

    def _percentile(self, data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    def stats(self) -> dict:
        with self._lock:
            if not self.latencies:
                return {"p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "mean": 0, "count": 0}

            return {
                "p50": round(self._percentile(self.latencies, 50), 2),
                "p95": round(self._percentile(self.latencies, 95), 2),
                "p99": round(self._percentile(self.latencies, 99), 2),
                "min": round(min(self.latencies), 2),
                "max": round(max(self.latencies), 2),
                "mean": round(statistics.mean(self.latencies), 2),
                "count": len(self.latencies),
            }


# ---------------------------------------------------------------------------
# Token Usage Tracker
# ---------------------------------------------------------------------------


class TokenTracker:
    """Track token usage by model and direction (input/output)."""

    def __init__(self):
        self.usage: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
        self._lock = threading.Lock()

    def record(self, model: str, input_tokens: int = 0, output_tokens: int = 0):
        with self._lock:
            self.usage[model]["input"] += input_tokens
            self.usage[model]["output"] += output_tokens

    def stats(self) -> dict:
        with self._lock:
            return {
                model: {
                    "input_tokens": v["input"],
                    "output_tokens": v["output"],
                    "total_tokens": v["input"] + v["output"],
                }
                for model, v in self.usage.items()
            }

    def total(self) -> int:
        with self._lock:
            return sum(v["input"] + v["output"] for v in self.usage.values())


# ---------------------------------------------------------------------------
# Cost Tracker
# ---------------------------------------------------------------------------


# USD per 1M tokens
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


class CostTracker:
    """Track costs per request and aggregate."""

    def __init__(self):
        self.costs: list[dict] = []
        self.total_cost = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def calculate(model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]

    def record(
        self,
        request_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: str | None = None,
    ):
        cost = self.calculate(model, input_tokens, output_tokens)
        entry = {
            "request_id": request_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "user_id": user_id,
            "timestamp": time.time(),
        }
        with self._lock:
            self.costs.append(entry)
            self.total_cost += cost
        return entry

    def stats(self) -> dict:
        with self._lock:
            if not self.costs:
                return {"total_cost_usd": 0, "avg_cost_usd": 0, "request_count": 0}

            by_model = defaultdict(float)
            by_user = defaultdict(float)
            for c in self.costs:
                by_model[c["model"]] += c["cost_usd"]
                if c["user_id"]:
                    by_user[c["user_id"]] += c["cost_usd"]

            return {
                "total_cost_usd": round(self.total_cost, 6),
                "avg_cost_usd": round(self.total_cost / len(self.costs), 6),
                "request_count": len(self.costs),
                "by_model": dict(by_model),
                "by_user": dict(by_user),
            }


# ---------------------------------------------------------------------------
# Error Rate Monitor
# ---------------------------------------------------------------------------


class ErrorRateMonitor:
    """Track error rates and types."""

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self.errors: list[dict] = []
        self.total_requests = 0
        self._lock = threading.Lock()

    def record_request(self):
        with self._lock:
            self.total_requests += 1

    def record_error(self, error_type: str, error_message: str, **kwargs):
        with self._lock:
            self.errors.append({
                "error_type": error_type,
                "error_message": error_message,
                "timestamp": time.time(),
                **kwargs,
            })

    def _cleanup(self):
        cutoff = time.time() - self.window_seconds
        self.errors = [e for e in self.errors if e["timestamp"] > cutoff]

    def stats(self) -> dict:
        with self._lock:
            self._cleanup()
            error_count = len(self.errors)
            error_rate = error_count / max(self.total_requests, 1)

            by_type = defaultdict(int)
            for e in self.errors:
                by_type[e["error_type"]] += 1

            return {
                "total_requests": self.total_requests,
                "error_count": error_count,
                "error_rate": round(error_rate, 4),
                "errors_by_type": dict(by_type),
            }


# ---------------------------------------------------------------------------
# Dashboard Data Generator
# ---------------------------------------------------------------------------


class DashboardData:
    """Generate aggregated data for monitoring dashboards."""

    def __init__(self):
        self.latency = LatencyTracker(window_size=10000)
        self.tokens = TokenTracker()
        self.costs = CostTracker()
        self.errors = ErrorRateMonitor(window_seconds=300)
        self.logger = StructuredLogger("monitoring")
        self.request_logger = RequestLogger(self.logger)

    def get_dashboard(self) -> dict:
        return {
            "timestamp": time.time(),
            "latency": self.latency.stats(),
            "tokens": self.tokens.stats(),
            "costs": self.costs.stats(),
            "errors": self.errors.stats(),
        }

    def print_dashboard(self):
        data = self.get_dashboard()
        print("\n" + "=" * 60)
        print("LLM SERVICE DASHBOARD")
        print("=" * 60)

        lat = data["latency"]
        print(f"\n  Latency (ms): p50={lat['p50']}  p95={lat['p95']}  p99={lat['p99']}  "
              f"min={lat['min']}  max={lat['max']}  mean={lat['mean']}  count={lat['count']}")

        tok = data["tokens"]
        if tok:
            print(f"\n  Token Usage:")
            for model, usage in tok.items():
                print(f"    {model}: input={usage['input_tokens']} output={usage['output_tokens']} "
                      f"total={usage['total_tokens']}")

        cost = data["costs"]
        print(f"\n  Cost: total=${cost['total_cost_usd']:.6f}  "
              f"avg=${cost['avg_cost_usd']:.6f}  requests={cost['request_count']}")
        if cost.get("by_model"):
            for model, c in cost["by_model"].items():
                print(f"    {model}: ${c:.6f}")

        err = data["errors"]
        print(f"\n  Errors: rate={err['error_rate']:.2%}  "
              f"count={err['error_count']}  total_requests={err['total_requests']}")
        if err.get("errors_by_type"):
            for etype, count in err["errors_by_type"].items():
                print(f"    {etype}: {count}")

        print("=" * 60)


# ---------------------------------------------------------------------------
# Alerting Thresholds
# ---------------------------------------------------------------------------


@dataclass
class AlertRule:
    name: str
    metric: str  # latency_p95, error_rate, cost_total, etc.
    threshold: float
    operator: str  # gt, lt, gte, lte
    severity: str  # warning, critical
    enabled: bool = True


@dataclass
class Alert:
    rule: AlertRule
    current_value: float
    timestamp: float
    message: str


class AlertingEngine:
    """Evaluate alert rules against current metrics."""

    def __init__(self):
        self.rules: list[AlertRule] = []
        self.alerts: list[Alert] = []
        self._lock = threading.Lock()

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def evaluate(self, dashboard: dict) -> list[Alert]:
        new_alerts = []
        for rule in self.rules:
            if not rule.enabled:
                continue

            value = self._extract_metric(dashboard, rule.metric)
            if value is None:
                continue

            triggered = False
            if rule.operator == "gt" and value > rule.threshold:
                triggered = True
            elif rule.operator == "lt" and value < rule.threshold:
                triggered = True
            elif rule.operator == "gte" and value >= rule.threshold:
                triggered = True
            elif rule.operator == "lte" and value <= rule.threshold:
                triggered = True

            if triggered:
                alert = Alert(
                    rule=rule,
                    current_value=value,
                    timestamp=time.time(),
                    message=f"[{rule.severity.upper()}] {rule.name}: "
                            f"{rule.metric}={value} (threshold: {rule.operator} {rule.threshold})",
                )
                new_alerts.append(alert)

        with self._lock:
            self.alerts.extend(new_alerts)
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]

        return new_alerts

    @staticmethod
    def _extract_metric(dashboard: dict, metric_path: str) -> float | None:
        parts = metric_path.split(".")
        current = dashboard
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return float(current) if isinstance(current, (int, float)) else None

    def get_alerts(self, severity: str | None = None) -> list[dict]:
        with self._lock:
            alerts = self.alerts
            if severity:
                alerts = [a for a in alerts if a.rule.severity == severity]
            return [
                {
                    "rule": a.rule.name,
                    "severity": a.rule.severity,
                    "metric": a.rule.metric,
                    "threshold": a.rule.threshold,
                    "current_value": a.current_value,
                    "message": a.message,
                    "timestamp": a.timestamp,
                }
                for a in alerts[-100:]  # Last 100
            ]


# ---------------------------------------------------------------------------
# Prometheus-Compatible Metrics (Text Format)
# ---------------------------------------------------------------------------


class PrometheusMetrics:
    """Generate Prometheus-compatible metrics in text format."""

    def __init__(self, prefix: str = "llm"):
        self.prefix = prefix
        self.counters: dict[str, float] = defaultdict(float)
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, list[float]] = defaultdict(list)

    def inc_counter(self, name: str, value: float = 1.0, **labels):
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        key = f"{self.prefix}_{name}{{{label_str}}}" if label_str else f"{self.prefix}_{name}"
        self.counters[key] += value

    def set_gauge(self, name: str, value: float, **labels):
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        key = f"{self.prefix}_{name}{{{label_str}}}" if label_str else f"{self.prefix}_{name}"
        self.gauges[key] = value

    def observe_histogram(self, name: str, value: float, **labels):
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        key = f"{self.prefix}_{name}{{{label_str}}}" if label_str else f"{self.prefix}_{name}"
        self.histograms[key].append(value)

    def render(self) -> str:
        lines = []

        for key, value in sorted(self.counters.items()):
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {value}")

        for key, value in sorted(self.gauges.items()):
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {value}")

        for key, values in sorted(self.histograms.items()):
            base = key.split("{")[0]
            lines.append(f"# TYPE {base} histogram")
            if values:
                lines.append(f'{key}_sum {sum(values)}')
                lines.append(f'{key}_count {len(values)}')

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo_monitoring():
    dashboard = DashboardData()

    # Simulate requests
    import random

    for i in range(50):
        model = random.choice(["gpt-4o", "gpt-4o-mini"])
        input_tokens = random.randint(100, 500)
        output_tokens = random.randint(50, 300)
        latency = random.uniform(200, 1500)

        dashboard.latency.record(latency)
        dashboard.tokens.record(model, input_tokens, output_tokens)
        dashboard.costs.record(
            request_id=f"req_{i}",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            user_id=f"user_{i % 5}",
        )
        dashboard.errors.record_request()

        if random.random() < 0.05:
            dashboard.errors.record_error(
                error_type="RateLimitError",
                error_message="Rate limit exceeded",
                model=model,
            )

    dashboard.print_dashboard()


def demo_alerting():
    engine = AlertingEngine()
    engine.add_rule(AlertRule(
        name="high_latency",
        metric="latency.p95",
        threshold=1000.0,
        operator="gt",
        severity="warning",
    ))
    engine.add_rule(AlertRule(
        name="critical_error_rate",
        metric="errors.error_rate",
        threshold=0.1,
        operator="gt",
        severity="critical",
    ))

    dashboard_data = {
        "latency": {"p50": 400, "p95": 1200, "p99": 3000},
        "errors": {"error_rate": 0.15, "total_requests": 100},
    }

    alerts = engine.evaluate(dashboard_data)
    print("\n=== Alerting Demo ===")
    for alert in alerts:
        print(f"  {alert.message}")

    if not alerts:
        print("  No alerts triggered")


def demo_prometheus():
    metrics = PrometheusMetrics()
    metrics.inc_counter("requests_total", model="gpt-4o", status="success")
    metrics.inc_counter("requests_total", model="gpt-4o", status="error")
    metrics.set_gauge("active_requests", 3)
    metrics.observe_histogram("request_duration_seconds", 0.5)
    metrics.observe_histogram("request_duration_seconds", 1.2)

    print("\n=== Prometheus Metrics ===")
    print(metrics.render())


if __name__ == "__main__":
    demo_monitoring()
    demo_alerting()
    demo_prometheus()
    print("\nAll monitoring demos complete.")
