from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = ROOT / "docker-compose.yml"


def _docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["docker", "compose", "version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _wait_for_port(host: str, port: int, timeout_seconds: float = 60.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(1.0)
    return False


@pytest.fixture(scope="session")
def kafka_stack() -> None:
    if os.getenv("RUN_KAFKA_INTEGRATION") != "1":
        pytest.skip("Kafka integration disabled. Set RUN_KAFKA_INTEGRATION=1.")

    if not _docker_available():
        pytest.skip("Docker/Docker Compose not available for Kafka integration tests.")

    try:
        import confluent_kafka  # type: ignore  # noqa: F401
    except Exception:
        pytest.skip("confluent_kafka not installed in this environment.")

    if _wait_for_port("localhost", 9092, timeout_seconds=2.0):
        yield
        return

    project = "nexus_alpha_kafka_it"
    up_cmd = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "-p",
        project,
        "up",
        "-d",
        "kafka",
    ]
    down_cmd = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "-p",
        project,
        "down",
        "-v",
    ]

    subprocess.run(up_cmd, check=True, cwd=ROOT)
    if not _wait_for_port("localhost", 9092):
        subprocess.run(down_cmd, check=False, cwd=ROOT)
        pytest.skip("Kafka port 9092 did not become ready in time.")

    yield

    subprocess.run(down_cmd, check=False, cwd=ROOT)
