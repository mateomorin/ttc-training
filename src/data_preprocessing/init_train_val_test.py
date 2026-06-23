"""
Step 0 — Initialize Shared Splits.
Run this once (locally, via console, ou dans un pod unique) pour générer 
les splits de validation et de test partagés.
"""

import logging
import hydra
from omegaconf import DictConfig
import polars as pl
import s3fs
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def get_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        endpoint_url="https://minio.lab.sspcloud.fr",
        client_kwargs={"region_name": "us-east-1"},
    )


def fetch_original_data(path: str, fs=None) -> pl.DataFrame:
    opener = fs.open if fs else open
    with opener(path) as f:
        df = pl.read_parquet(f)
    df = df.rename(mapping={"nace2025": "code", "libelle": "label"})[["code", "label"]]
    df = df.with_columns(
        (pl.col("code").str.slice(0, 2) + "." + pl.col("code").str.slice(2)).alias("code")
    )
    return df


def split_guaranteed_and_remaining(
    df: pl.DataFrame,
    code_column: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Sépare le dataset en deux :
    - Un dataframe garanti contenant exactement une ligne par code unique (mélangé proprement).
    - Un dataframe contenant le reste des lignes disponibles.
    """
    # On mélange d'abord pour ne pas toujours prendre la première ligne absolue du fichier source
    df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)

    df_guaranteed = df_shuffled.unique(subset=[code_column], keep="first", maintain_order=True)
    df_remaining = df_shuffled.join(
        df_guaranteed,
        on=df.columns,
        how="anti",
        maintain_order="left_right"
    )

    return df_guaranteed, df_remaining


@hydra.main(version_base=None, config_path="../../config", config_name="data_config")
def main(cfg: DictConfig) -> None:
    fs = get_fs()
    val_sample = int(cfg.val_sample)
    train_sample = int(cfg.final_size)
    output_prefix = cfg.output_prefix

    train_key = f"{output_prefix}/shared/train_n{train_sample}.parquet"
    val_key = f"{output_prefix}/shared/val_n{val_sample}.parquet"
    test_key = f"{output_prefix}/shared/test.parquet"

    logger.info("Checking shared splits...")

    # Train
    if not fs.exists(train_key):
        logger.info(f"Generating shared train split → {train_key}")
        df_train = fetch_original_data(cfg.original_train_path, fs)
        df_train_base, df_train_rem = split_guaranteed_and_remaining(df_train, "code")
        rem_sample = train_sample - len(df_train_base)
        if rem_sample < 0:
            logger.warn("The sample is too small to have all codes in train, skipping, ...")
            return
        df_train = pl.concat([df_train_base, df_train_rem.head(rem_sample)])
        df_train = df_train.sample(fraction=1.0, seed=42, shuffle=True)
        with fs.open(train_key, "wb") as f:
            df_train.write_parquet(f)
    else:
        logger.info("Train split already exists.")

    # Validation
    if not fs.exists(val_key):
        logger.info(f"Generating shared validation split → {val_key}")
        df_val = fetch_original_data(cfg.original_val_path, fs)
        df_val = df_val.sample(n=val_sample, shuffle=True, seed=42)
        with fs.open(val_key, "wb") as f:
            df_val.write_parquet(f)
    else:
        logger.info("Validation split already exists.")

    # Test
    if not fs.exists(test_key):
        logger.info(f"Generating shared test split → {test_key}")
        df_test = fetch_original_data(cfg.original_test_path, fs)
        df_test = df_test.sample(fraction=1, shuffle=True, seed=42)
        with fs.open(test_key, "wb") as f:
            df_test.write_parquet(f)
    else:
        logger.info("Test split already exists.")

    logger.info(f"Initialization complete ! Train, validation, and test datasets are sampled and shuffled in folder {output_prefix}/shared/")


if __name__ == "__main__":
    main()
