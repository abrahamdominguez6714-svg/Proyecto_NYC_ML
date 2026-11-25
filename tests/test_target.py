"""Pruebas del target P_24h y de las variables predictoras.   uv run pytest"""
import pandas as pd

from src.config import FEATURES, FORBIDDEN_FEATURES, TARGET
from src.target import make_target


def _raw(rows):
    return pd.DataFrame(rows, columns=["unique_key", "created_date", "closed_date", "complaint_type",
                                       "borough", "open_data_channel_type", "location_type"])


def test_target_definicion_y_filtros():
    base = "2025-01-10T08:00:00"
    df = _raw([
        ("1", base, "2025-01-10T08:01:00", "A", "X", "PHONE", "S"),   # 60 s   -> descartada (<= 150 s)
        ("2", base, "2025-01-10T10:00:00", "A", "X", "PHONE", "S"),   # 2 h    -> 1
        ("3", base, "2025-01-11T07:59:00", "A", "X", "PHONE", "S"),   # 23h59  -> 1
        ("4", base, "2025-01-11T08:00:00", "A", "X", "PHONE", "S"),   # 24 h   -> 0 (el límite cuenta como lento)
        ("5", base, "2025-01-14T08:00:00", "A", "X", "PHONE", "S"),   # 4 días -> 0
        ("6", base, None, "A", "X", "PHONE", "S"),                    # sin cierre -> descartada por defecto
    ])
    out = make_target(df)
    assert out["unique_key"].tolist() == ["2", "3", "4", "5"]
    assert out[TARGET].tolist() == [1, 1, 0, 0]
    assert {"hour", "weekday"} <= set(out.columns)
    assert "Diff" not in df.columns           # no muta el DataFrame de entrada


def test_target_conserva_no_cerradas_si_se_pide():
    df = _raw([("1", "2025-01-10T08:00:00", None, "A", "X", "PHONE", "S")])
    out = make_target(df, keep_unclosed=True)
    assert out[TARGET].tolist() == [0]


def test_features_no_incluyen_variables_prohibidas():
    assert not (set(FEATURES) & FORBIDDEN_FEATURES)
