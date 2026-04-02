from scripts.run_backtest_matrix import _extract_total_metrics


def test_extract_total_metrics_parses_backtest_report() -> None:
    sample = """
    Result for strategy NexusAlphaStrategy
                                                 BACKTESTING REPORT
    ┏━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃     Pair ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃ Avg Duration ┃  Win  Draw  Loss  Win% ┃
    ┡━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ BTC/USDT │      1 │         2.50 │          25.000 │         0.25 │         2:00 │    1     0     0   100 │
    │    TOTAL │      3 │         1.50 │          45.000 │         0.45 │         4:00 │    2     0     1    67 │
    └──────────┴────────┴──────────────┴─────────────────┴──────────────┴──────────────┴────────────────────────┘
    """

    metrics = _extract_total_metrics(sample)

    assert metrics == {
        "trades": 3,
        "avg_profit_pct": 1.5,
        "total_profit_usdt": 45.0,
        "total_profit_pct": 0.45,
    }
