import pandas as pd
from datetime import datetime, timedelta
import train_model


def test_time_split_order_and_no_overlap():
    # create a chronological dataframe with 100 rows
    base = datetime(2025, 1, 1)
    rows = []
    for i in range(100):
        rows.append({'date': (base + timedelta(days=i)), 'target': i % 2, 'surface': 'Hard', 'player': 'p', 'opponent': 'q'})
    df = pd.DataFrame(rows)

    train, val, test = train_model.time_split(df, train_frac=0.7, val_frac=0.15)

    # Ensure chronological
    assert train['date'].max() <= val['date'].min()
    assert val['date'].max() <= test['date'].min()

    # Sizes roughly
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
