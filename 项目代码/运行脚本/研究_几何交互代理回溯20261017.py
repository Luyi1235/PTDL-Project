"""Compare frozen models on previously inspected holdouts, explicitly retrospective."""

import json
import time

import numpy as np
import pandas as pd

import 研究_几何交互代理20261017 as study


def main():
    models = {
        "geometry_interaction": study.OUT / "几何交互候选代理.json",
        "previous1512": study.prior.OUT / "多精度候选代理.json",
        "workbench289": study.prior.original.prior.driver.LOW_MODEL,
    }
    loaded = {key: study.prior.direct.base.TargetwiseAeroGP.from_json(path)
              for key, path in models.items()}
    rows = []
    for batch, folder in (("earlier96", study.prior.SOURCE), ("later96", study.prior.OUT)):
        frame = pd.read_csv(folder / "中网格独立测试标签.csv")
        for name, model in loaded.items():
            predictions = np.asarray([
                model.predict(study.prior.original.prior.AirframeDesign(
                    **{key: float(row[key]) for key in study.FEATURES}))
                for _, row in frame.iterrows()])
            for index, target in enumerate(study.TARGETS):
                y = frame[target].to_numpy(float)
                valid = np.isfinite(y)
                if index:
                    valid &= frame.wake_quality_pass.to_numpy(bool)
                frame[f"{name}_prediction_{index}"] = predictions[:, index]
                rows.append(dict(batch=batch, model=name, target=target, count=int(valid.sum()),
                                 **study.prior.direct.base.metric(y[valid], predictions[valid, index])))
        frame.to_csv(study.OUT / f"{batch}_回溯逐点.csv", index=False)
    result = dict(metrics=rows, retrospective=True, used_for_model_selection=False,
                  fresh_validation_required=True, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    (study.OUT / "历史构型回溯评价.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    pd.DataFrame(rows).to_csv(study.OUT / "历史构型回溯评价.csv", index=False)
    print(pd.DataFrame(rows)[["batch", "model", "target", "r2", "mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
