from __future__ import annotations

import pandas as pd
import pytest

from models import predict_next_7_days


def test_predict_next_7_days_rejects_unknown_model() -> None:
    prepared = pd.DataFrame({"pm2_5": [10.0] * 40})
    with pytest.raises(ValueError):
        predict_next_7_days("unknown", prepared)
