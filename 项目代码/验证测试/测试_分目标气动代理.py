import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"核心模型"), str(ROOT/"运行脚本")]
from 分目标气动高斯过程 import FEATURES, TargetwiseAeroGP, feature_map, reconstructed_ew
from 运行_分目标气动代理攻关 import fit_target, annotate_quality, TRAIN
from VSPAERO气动代理模型 import load_vsp_surrogate
from 机体参数化设计与气动阻力 import AirframeDesign


class TargetwiseGPTests(unittest.TestCase):
    def test_quality_annotation_is_idempotent(self):
        import pandas as pd
        frame=pd.read_csv(TRAIN).head(1)
        first=annotate_quality(frame)
        second=annotate_quality(first)
        pd.testing.assert_frame_equal(first,second)

    def test_polar_model_roundtrip_and_invalid_polar_fallback(self):
        import copy
        base=dict(features="raw",kernel="rbf",feature_mean=[0.]*7,feature_scale=[1.]*7,
                  train=[[0.]*7],mean=0.,scale=1.,amplitude=1.,length=[1.]*7,
                  noise=.01,dual=[0.],inverse=[[1.]],interval_scale=1.)
        ar=AirframeDesign().aspect_ratio
        coeff=np.concatenate(([0.,.1,.2],[0.,.1,.2],np.array([.01,.04,.04])/(np.pi*ar*.8)))
        components=[base|{"mean":float(c)} for c in coeff]
        polar=dict(kind="polar",components=components,fallback=base|{"mean":.7},
                   interval_std=.03,train_count=1)
        payload=dict(schema="ptdl_targetwise_gp_v1",feature_labels=list(FEATURES),
                     training_samples=1,baseline_targets=[6.,.8],targets=[base|{"mean":6.},polar])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"polar.json"
            TargetwiseAeroGP(payload).to_json(path)
            model=load_vsp_surrogate(path)
            self.assertAlmostEqual(model.predict(AirframeDesign())[1],.8)
            broken=copy.deepcopy(payload)
            for component in broken["targets"][1]["components"][6:]:
                component["mean"]*=-1
            fallback=TargetwiseAeroGP(broken)
            self.assertAlmostEqual(fallback.predict(AirframeDesign())[1],.7)
            self.assertGreaterEqual(fallback.predict_with_uncertainty(AirframeDesign())[3],.03)

    def test_polar_reconstruction_keeps_total_lift_wake_drag_definition(self):
        ar=24.
        cl=np.array([0.,.1,.2])
        cd=np.array([.01,.04,.04])/(np.pi*ar*.8)
        coeff=np.concatenate([cl,cl,cd])
        self.assertAlmostEqual(reconstructed_ew(coeff,ar),.8)
        coeff[6:]*=-1
        self.assertTrue(np.isnan(reconstructed_ew(coeff,ar)))

    def test_dimensionless_map_respects_uniform_length_scaling(self):
        x = np.array([[26,24,.45,1,0,8.38,.75]])
        scaled = x.copy()
        scaled[:,0] *= 4
        scaled[:,5:] *= 2
        np.testing.assert_allclose(feature_map(x,"dimensionless"), feature_map(scaled,"dimensionless"))

    def test_invalid_features_rejected(self):
        with self.assertRaises(ValueError):
            feature_map(np.zeros((1,7)),"dimensionless")
        with self.assertRaises(ValueError):
            feature_map(np.ones((1,7)),"unknown")

    def test_portable_gp_matches_sklearn_all_kernels(self):
        rng=np.random.default_rng(61)
        base=np.array([getattr(AirframeDesign(),k) for k in FEATURES])
        x=base+rng.normal(size=(18,7))*[2,2,.04,.2,.2,.3,.03]
        y=6+.04*x[:,1]-.01*x[:,4]+.003*rng.normal(size=len(x))
        design=AirframeDesign()
        for mode in ("raw","dimensionless"):
            for kind in ("rbf","matern32","matern52"):
                with self.subTest(mode=mode,kind=kind):
                    gp,params=fit_target(x,y,mode,kind)
                    payload=dict(schema="ptdl_targetwise_gp_v1",feature_labels=list(FEATURES),
                                 targets=[params,params],training_samples=len(x),baseline_targets=[6,.7])
                    model=TargetwiseAeroGP(payload)
                    z=(feature_map(base,mode)-params["feature_mean"])/params["feature_scale"]
                    expected,std=gp.predict(z,return_std=True)
                    got=model.predict_with_uncertainty(design)
                    np.testing.assert_allclose(got[:2],expected[0],rtol=1e-9)
                    np.testing.assert_allclose(got[2:],std[0],rtol=2e-5,atol=1e-7)
                    with tempfile.TemporaryDirectory() as folder:
                        path=Path(folder)/"model.json"
                        model.to_json(path)
                        restored=load_vsp_surrogate(path)
                        np.testing.assert_allclose(restored.predict_with_uncertainty(design),got)
                        broken=json.loads(path.read_text(encoding="utf-8"))
                        broken["targets"][0]["length"][0]=-1
                        path.write_text(json.dumps(broken),encoding="utf-8")
                        with self.assertRaises(ValueError):
                            load_vsp_surrogate(path)


if __name__=="__main__":
    unittest.main()
