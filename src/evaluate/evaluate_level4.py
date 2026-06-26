"""
Évaluation niveau 4 — tous les runs de l'experiment MLflow "8".

Pour chaque run :
  - Charge le modèle PyTorch + tokenizer + value_encoder depuis les artefacts
  - Prédit sur le même jeu de test qu'à l'entraînement
  - Calcule toutes les métriques au niveau 5 (codes originaux) ET au niveau 4
    (code tronqué : DD.DDL → DD.DD)
  - Calcule les métriques par zone Head / Body / Tail (définies sur X_train_real)
  - Logge tout dans un nouveau run de l'experiment "benchmark-evaluation-level4"

Usage
-----
    python evaluate_level4.py

    # Surcharger les paramètres par défaut :
    python evaluate_level4.py \
        --experiment_id   8 \
        --val_test_path   s3://mateom/graal/ttc-injection/v2/shared \
        --zones_path      s3://mateom/graal/distribution_zones.npz
"""

import logging
import os
import pickle

import hydra
import mlflow
import numpy as np
import polars as pl
import s3fs
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score
from torchTextClassifiers import ModelConfig, torchTextClassifiers

load_dotenv(override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

CONFIDENCE_THRESHOLD = 0.70
EVAL_EXPERIMENT_NAME = "benchmark-evaluation-level4"


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def get_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        endpoint_url="https://minio.lab.sspcloud.fr",
        client_kwargs={"region_name": "us-east-1"},
    )


def fetch_parquet(path: str, fs: s3fs.S3FileSystem) -> pl.DataFrame:
    with fs.open(path) as f:
        return pl.read_parquet(f)


def load_test_data(
    val_test_path: str,
    preprocessed: bool,
    fs: s3fs.S3FileSystem,
) -> pl.DataFrame:
    pre = "_preprocessed" if preprocessed else ""
    path = f"{val_test_path.strip('/')}/test{pre}.parquet"
    logger.info(f"Chargement test : {path}")
    return fetch_parquet(path, fs)


# ---------------------------------------------------------------------------
# Niveau 4 : DD.DDL → DD.DD
# ---------------------------------------------------------------------------

def to_level4(codes: np.ndarray) -> np.ndarray:
    """Retire le dernier caractère (lettre) pour remonter au niveau 4."""
    return np.array([c[:-1] if len(c) > 0 else c for c in codes])


# ---------------------------------------------------------------------------
# Chargement du modèle depuis les artefacts du run
# ---------------------------------------------------------------------------

def load_pipeline_from_run(
    run_id: str,
    client: MlflowClient,
    n_classes: int,
    embedding_dim: int
) -> torchTextClassifiers:
    """
    Charge le modèle PyTorch depuis artifacts/model/ et le tokenizer +
    value_encoder depuis artifacts/pipeline/.
    """
    # — Modèle PyTorch —
    model_uri = f"runs:/{run_id}/model"
    logger.info(f"  Chargement modèle : {model_uri}")
    pytorch_model = mlflow.pytorch.load_model(model_uri)

    # — Tokenizer & ValueEncoder —
    tokenizer     = None
    value_encoder = None

    local_pipeline_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="pipeline"
    )

    with open(os.path.join(local_pipeline_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(local_pipeline_dir, "value_encoder.pkl"), "rb") as f:
        value_encoder = pickle.load(f)

    if tokenizer is None or value_encoder is None:
        raise RuntimeError(
            f"Impossible de récupérer tokenizer/value_encoder pour run {run_id} "
            "et aucun fallback disponible."
        )

    model_config = ModelConfig(
        embedding_dim=embedding_dim,
        num_classes=n_classes,
    )
    ttc = torchTextClassifiers(
        tokenizer=tokenizer,
        model_config=model_config,
        value_encoder=value_encoder,
    )
    ttc.pytorch_model = pytorch_model
    ttc.pytorch_model.eval()
    return ttc


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def compute_metrics_at_level(
    y_true: np.ndarray,
    preds_top5: np.ndarray,
    conf_top1: np.ndarray,
    level: int,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Calcule les métriques globales pour un niveau hiérarchique donné (4 ou 5).
    Au niveau 4, y_true et preds_top5 sont déjà tronqués avant l'appel.
    """
    preds_top1 = preds_top5[:, 0]

    acc1 = (preds_top1 == y_true).mean()
    acc3 = np.any(preds_top5[:, :3] == y_true[:, None], axis=1).mean()
    acc5 = np.any(preds_top5        == y_true[:, None], axis=1).mean()

    confident_mask = conf_top1 > threshold
    coverage       = confident_mask.mean()
    acc_conf       = (
        (preds_top1[confident_mask] == y_true[confident_mask]).mean()
        if confident_mask.sum() > 0 else 0.0
    )

    f1_macro    = f1_score(y_true, preds_top1, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, preds_top1, average="weighted", zero_division=0)

    prefix = f"lvl{level}"
    return {
        f"{prefix}_acc_top1":        float(acc1),
        f"{prefix}_acc_top3":        float(acc3),
        f"{prefix}_acc_top5":        float(acc5),
        f"{prefix}_f1_macro":        float(f1_macro),
        f"{prefix}_f1_weighted":     float(f1_weighted),
        f"{prefix}_coverage_rate":   float(coverage),
        f"{prefix}_acc_confident":   float(acc_conf),
    }


def compute_zone_metrics(
    y_true: np.ndarray,
    preds_top1: np.ndarray,
    preds_top5: np.ndarray,
    conf_top1: np.ndarray,
    head: np.ndarray,
    body: np.ndarray,
    tail: np.ndarray,
    level: int,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Métriques par zone Head / Body / Tail pour un niveau hiérarchique donné.
    Les zones sont toujours définies sur les codes du niveau considéré.
    """
    metrics = {}
    for zone_name, zone_codes in [("head", head), ("body", body), ("tail", tail)]:
        mask = np.isin(y_true, zone_codes)
        n    = mask.sum()
        prefix = f"lvl{level}_{zone_name}"

        if n == 0:
            logger.warning(f"  [lvl{level}] Zone {zone_name} : aucun exemple dans le test.")
            metrics.update({
                f"{prefix}_n_samples":     0,
                f"{prefix}_n_codes_seen":  0,
                f"{prefix}_acc_top1":      0.0,
                f"{prefix}_acc_top5":      0.0,
                f"{prefix}_f1_macro":      0.0,
                f"{prefix}_coverage_rate": 0.0,
                f"{prefix}_acc_confident": 0.0,
            })
            continue

        yt   = y_true[mask]
        p1   = preds_top1[mask]
        p5   = preds_top5[mask]
        conf = conf_top1[mask]

        acc1           = (p1 == yt).mean()
        acc5           = np.any(p5 == yt[:, None], axis=1).mean()
        f1             = f1_score(yt, p1, average="macro", zero_division=0)
        n_codes_seen   = len(np.unique(yt))
        confident_mask = conf > threshold
        coverage       = confident_mask.mean()
        acc_conf       = (
            (p1[confident_mask] == yt[confident_mask]).mean()
            if confident_mask.sum() > 0 else 0.0
        )

        logger.info(
            f"  [lvl{level}] Zone {zone_name:4s} | n={n:7d} | "
            f"codes vus={n_codes_seen:3d}/{len(zone_codes):3d} | "
            f"Acc@1={acc1:.4f} | Acc@5={acc5:.4f} | F1={f1:.4f} | "
            f"Coverage={coverage:.4f} | Acc@conf>{int(threshold*100)}%={acc_conf:.4f}"
        )
        metrics.update({
            f"{prefix}_n_samples":     int(n),
            f"{prefix}_n_codes_seen":  int(n_codes_seen),
            f"{prefix}_acc_top1":      float(acc1),
            f"{prefix}_acc_top5":      float(acc5),
            f"{prefix}_f1_macro":      float(f1),
            f"{prefix}_coverage_rate": float(coverage),
            f"{prefix}_acc_confident": float(acc_conf),
        })

    return metrics


def build_zones(
    zones_path: str,
    fs: s3fs.S3FileSystem,
    level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit les zones Head / Body / Tail sauvegardés.
    Si level=4, les codes sont d'abord tronqués au niveau 4 avant le ranking.
    """
    if level == 4:
        zones_path = zones_path[:-4] + "_lvl4.npz"
    opener = fs.open if fs else open
    with opener(zones_path, "rb") as f:
        with np.load(f, allow_pickle=True) as data:
            head = data["head"]
            body = data["body"]
            tail = data["tail"]
    return head, body, tail


# ---------------------------------------------------------------------------
# Listing des runs de l'experiment
# ---------------------------------------------------------------------------

def get_runs_from_experiment(
    experiment_id: str,
    client: MlflowClient,
    select_run: str = None,
) -> list[mlflow.entities.Run]:
    """
    Retourne tous les runs FINISHED de l'experiment, triés par start_time.
    Les runs en erreur ou en cours sont ignorés.
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    )
    logger.info(f"{len(runs)} runs FINISHED trouvés dans l'experiment {experiment_id}.")

    if select_run is not None:
        runs = [run for run in runs if run.info.run_id == select_run]
        logger.info(f"Sélection de la run {select_run} dans l'expérience {experiment_id}.")
    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../config", config_name="evaluate_config")
def main(cfg) -> None:
    client = MlflowClient()
    fs     = get_fs()

    # ------------------------------------------------------------------
    # 1. Données
    # ------------------------------------------------------------------
    df_test  = load_test_data(
        cfg.val_test_path, cfg.preprocessed, fs
    )

    X_test  = df_test["label"].to_numpy()
    y_test  = df_test["code"].to_numpy()

    n_classes_lvl5 = len(np.unique(y_test))
    logger.info(f"Test : {len(X_test)} exemples | {n_classes_lvl5} classes niveau 5")

    # Zones niveau 5
    head5, body5, tail5 = build_zones(cfg.zones_path, fs, level=5)
    logger.info(
        f"Zones lvl4 — Head: {len(head5)} | Body: {len(body5)} | Tail: {len(tail5)}"
    )

    # Zones niveau 4
    y_test_lvl4 = to_level4(y_test)
    n_classes_lvl4 = len(np.unique(y_test_lvl4))
    logger.info(f"Niveau 4 : {n_classes_lvl4} codes uniques dans le test")

    head4, body4, tail4 = build_zones(cfg.zones_path, fs, level=4)
    logger.info(
        f"Zones lvl4 — Head: {len(head4)} | Body: {len(body4)} | Tail: {len(tail4)}"
    )

    # ------------------------------------------------------------------
    # 3. Récupération des runs
    # ------------------------------------------------------------------
    runs = get_runs_from_experiment(cfg.experiment_id, client, cfg.select_run)

    # ------------------------------------------------------------------
    # 4. Boucle d'évaluation
    # ------------------------------------------------------------------
    mlflow.set_experiment(EVAL_EXPERIMENT_NAME)

    for run in runs:
        run_id   = run.info.run_id
        run_name = run.data.tags.get("mlflow.runName", run_id[:8])
        params   = run.data.params

        logger.info(f"\n{'='*65}")
        logger.info(f"Run : {run_name}  ({run_id})")
        logger.info(f"{'='*65}")

        # Récupère l'embedding_dim du run source pour reconstruire le bon modèle
        embedding_dim = int(params.get("model.embedding_dim", 32))

        try:
            ttc = load_pipeline_from_run(
                run_id=run_id,
                client=client,
                n_classes=n_classes_lvl5,
                embedding_dim=embedding_dim
            )
        except Exception as e:
            logger.error(f"  Impossible de charger le modèle : {e} — run ignoré.")
            continue

        # — Prédictions niveau 5 —
        logger.info("  Prédictions …")
        results    = ttc.predict(X_test, top_k=5)
        preds_top5 = np.array(results["prediction"])
        conf_top5  = np.array(
            [[c.item() if hasattr(c, "item") else c for c in row]
             for row in results["confidence"]]
        )
        conf_top1  = conf_top5[:, 0]
        preds_top1_lvl5 = preds_top5[:, 0]

        # — Prédictions niveau 4 (tronquer les sorties) —
        y_test_lvl4     = to_level4(y_test)
        preds_top5_lvl4 = np.vectorize(lambda c: c[:-1] if len(c) > 0 else c)(preds_top5)
        preds_top1_lvl4 = preds_top5_lvl4[:, 0]

        # — Métriques globales —
        global_lvl5 = compute_metrics_at_level(y_test,      preds_top5,      conf_top1, level=5)
        global_lvl4 = compute_metrics_at_level(y_test_lvl4, preds_top5_lvl4, conf_top1, level=4)

        logger.info(
            f"  [lvl5] Acc@1={global_lvl5['lvl5_acc_top1']:.4f} | "
            f"F1={global_lvl5['lvl5_f1_macro']:.4f}"
        )
        logger.info(
            f"  [lvl4] Acc@1={global_lvl4['lvl4_acc_top1']:.4f} | "
            f"F1={global_lvl4['lvl4_f1_macro']:.4f}"
        )

        # — Métriques par zone —
        zone_lvl5 = compute_zone_metrics(
            y_true=y_test,           preds_top1=preds_top1_lvl5,
            preds_top5=preds_top5,   conf_top1=conf_top1,
            head=head5, body=body5,  tail=tail5, level=5,
        )
        zone_lvl4 = compute_zone_metrics(
            y_true=y_test_lvl4,          preds_top1=preds_top1_lvl4,
            preds_top5=preds_top5_lvl4,  conf_top1=conf_top1,
            head=head4, body=body4,      tail=tail4, level=4,
        )

        # — Log MLflow dans l'experiment d'évaluation —
        with mlflow.start_run(run_name=f"eval_lvl4__{run_name}"):

            # Paramètres du run source pour traçabilité
            mlflow.log_params({
                "source_run_id":      run_id,
                "source_run_name":    run_name,
                "source_synth_name":  params.get("input_data.synth_name",  "baseline"),
                "source_synth_split": params.get("input_data.synth_split", "N/A"),
                "source_embedding_dim": embedding_dim,
                "eval_test_size":     len(X_test),
                "confidence_threshold": CONFIDENCE_THRESHOLD,
            })

            mlflow.log_metrics({
                **global_lvl5,
                **global_lvl4,
                **zone_lvl5,
                **zone_lvl4,
            })

        logger.info(f"  → Métriques loguées dans '{EVAL_EXPERIMENT_NAME}'.")

    logger.info("\nÉvaluation terminée.")


if __name__ == "__main__":
    main()