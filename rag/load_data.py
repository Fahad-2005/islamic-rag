from pathlib import Path
import pandas as pd


DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "banuri_scraped.csv"
)


def load_fatwas():
    df = pd.read_csv(
        DATA_FILE,
        encoding="utf-8-sig"
    )

    print(f"Loaded records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    df = load_fatwas()

    print("\nFirst record:")
    print(df.iloc[0].to_dict())