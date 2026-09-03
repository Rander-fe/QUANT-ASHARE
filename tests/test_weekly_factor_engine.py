import pandas as pd

from backtest.weekly_factor_engine import weekly_schedule


def test_weekly_schedule_uses_last_actual_trading_day_and_next_open_day():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"])
    result = weekly_schedule(dates, pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-11"))
    assert result.iloc[0].to_dict() == {
        "signal_date": pd.Timestamp("2024-01-05"),
        "execution_date": pd.Timestamp("2024-01-08"),
    }


def test_weekly_schedule_handles_friday_holiday():
    dates = pd.to_datetime(["2024-04-01", "2024-04-02", "2024-04-03",
                            "2024-04-08", "2024-04-09"])
    result = weekly_schedule(dates, pd.Timestamp("2024-04-01"), pd.Timestamp("2024-04-09"))
    assert result.iloc[0]["signal_date"] == pd.Timestamp("2024-04-03")
    assert result.iloc[0]["execution_date"] == pd.Timestamp("2024-04-08")
