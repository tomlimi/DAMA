from pathlib import Path

import yaml

with open("globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, OUTPUT_DIR, DATA_DIR, STATS_DIR, MODEL_DIR, HPARAMS_DIR,) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["OUTPUT_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["MODEL_DIR"],
        data["HPARAMS_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
