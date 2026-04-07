#!/usr/bin/env python3
"""TimescaleDB maintenance utilities.

This script will detect hypertables in the connected TimescaleDB instance and
optionally run drop_chunks to remove old data and run a basic VACUUM/FULL.

Usage:
  python infra/maintenance/timescale_maintenance.py --retention-days 90 --dry-run

It reads TIMESCALEDB_URL from environment or .env (via NexusConfig). It tries to
use psycopg (psycopg3 or psycopg2) when available, otherwise falls back to
invoking psql if the CLI is available.

NOTE: This script is conservative: by default it only prints actions (dry-run).
Use --execute to actually run drop_chunks.
"""

from __future__ import annotations

import os
import subprocess
import sys
from urllib.parse import urlparse
from typing import List

import click

from nexus_alpha.config import NexusConfig, load_config
from nexus_alpha.logging import get_logger

logger = get_logger(__name__)


def _get_db_url(env_file: str | None = None) -> str:
    cfg = load_config(env_file or ".env")
    return cfg.database.timescaledb_url.get_secret_value()


def _psql_available() -> bool:
    from shutil import which

    return which("psql") is not None


def _run_psql(sql: str, db_url: str) -> int:
    parsed = urlparse(db_url.replace("postgresql+asyncpg://", "postgresql://", 1))
    user = parsed.username or "postgres"
    pw = parsed.password or ""
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    dbname = parsed.path.lstrip("/") or "postgres"

    env = os.environ.copy()
    if pw:
        env["PGPASSWORD"] = pw

    cmd = [
        "psql",
        f"-h", host,
        f"-p", str(port),
        f"-U", user,
        f"-d", dbname,
        f"-c", sql,
    ]
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


@click.command()
@click.option("--retention-days", default=90, type=int, help="Retention window in days")
@click.option("--dry-run/--execute", default=True, help="Print actions or actually execute")
@click.option("--env-file", default=None, help="Path to .env file to load DB URL")
def main(retention_days: int, dry_run: bool, env_file: str | None) -> None:
    """Detect hypertables and drop chunks older than retention window."""
    db_url = _get_db_url(env_file)
    click.echo(f"Using DB URL: {db_url}")
    sql_list = []

    # Query hypertables
    query_ht = (
        "SELECT hypertable_schema, hypertable_name FROM timescaledb_information.hypertables;"
    )

    if _psql_available():
        click.echo("psql available — using psql to enumerate hypertables")
        # Fetch list via psql
        cmd = [
            "psql",
            "-At",
            "-c",
            query_ht,
        ]
        parsed = urlparse(db_url.replace("postgresql+asyncpg://", "postgresql://", 1))
        if parsed.username:
            cmd.extend(["-U", parsed.username])
        if parsed.hostname:
            cmd.extend(["-h", parsed.hostname])
        if parsed.port:
            cmd.extend(["-p", str(parsed.port)])
        if parsed.path and parsed.path != "/":
            cmd.extend(["-d", parsed.path.lstrip("/")])
        env = os.environ.copy()
        if parsed.password:
            env["PGPASSWORD"] = parsed.password
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
        if proc.returncode != 0:
            click.echo("Failed to query hypertables via psql:\n" + proc.stderr)
            sys.exit(1)
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            schema, name = line.split("|") if "|" in line else ("public", line.strip())
            sql_list.append((schema, name))
    else:
        click.echo("psql not available — attempting to use psycopg (if installed)")
        try:
            import psycopg

            conn = psycopg.connect(_get_db_url(env_file))
            cur = conn.cursor()
            cur.execute(query_ht)
            for row in cur.fetchall():
                sql_list.append((row[0], row[1]))
            cur.close()
            conn.close()
        except Exception as err:
            click.echo(f"Failed to enumerate hypertables: {err}")
            sys.exit(1)

    click.echo(f"Detected hypertables: {sql_list}")
    if not sql_list:
        click.echo("No hypertables detected — nothing to do.")
        return

    for schema, name in sql_list:
        drop_sql = f"SELECT drop_chunks(interval '{retention_days} days', '{schema}.{name}');"
        vacuum_sql = f"VACUUM (VERBOSE, ANALYZE) {schema}.{name};"
        click.echo(f"Action for {schema}.{name}:\n  {drop_sql}\n  {vacuum_sql}\n")
        if not dry_run:
            click.echo(f"Executing drop_chunks for {schema}.{name}...")
            rc = _run_psql(drop_sql, db_url)
            if rc != 0:
                click.echo(f"drop_chunks failed for {schema}.{name}, exit {rc}")
            else:
                click.echo("drop_chunks finished — running VACUUM")
                rc2 = _run_psql(vacuum_sql, db_url)
                if rc2 != 0:
                    click.echo(f"VACUUM failed for {schema}.{name}, exit {rc2}")
                else:
                    click.echo("VACUUM completed")

    click.echo("Maintenance complete.")


if __name__ == "__main__":
    main()
