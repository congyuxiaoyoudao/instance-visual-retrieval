"""Provided helpers should pass before students implement any TODO."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from instance_retrieval.data import RevisitedDataset
from instance_retrieval.evaluation import evaluate
from instance_retrieval.vendor.evaluate import compute_map
from instance_retrieval.utils import normalize, save_npz
pytestmark=pytest.mark.infrastructure

def test_dataset_query_crop(tmp_path):
    directory=tmp_path/"roxford5k"
    (directory/"jpg").mkdir(parents=True)
    Image.new("RGB",(80,60),"white").save(directory/"jpg/a.jpg")
    cfg={"imlist":["a"],"qimlist":["a"],
         "gnd":[{"bbx":[10,20,50,55],"easy":[],"hard":[0],"junk":[]}]}
    (directory/"gnd_roxford5k.pkl").write_bytes(pickle.dumps(cfg))
    ds=RevisitedDataset(tmp_path,"roxford5k")
    assert ds.image(0).size == (80,60)
    assert ds.image(0,query=True).size == (40,35)

def test_evaluator_upstream_integrity():
    import instance_retrieval.vendor.evaluate as upstream
    source=Path(upstream.__file__)
    provenance=json.loads(source.with_name("provenance.json").read_text(encoding="utf-8-sig"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance["sha256"]

def test_official_evaluator_known_ranking():
    mean_ap,_,precision,_=compute_map(np.array([[0],[1],[2],[3]]),
                                     [{"ok":[1,3],"junk":[0]}],[10])
    assert mean_ap == pytest.approx(19/24)
    assert precision[0] == pytest.approx(2/3)

def test_duplicate_ranks_rejected_before_student_protocol():
    with pytest.raises(ValueError,match="permutation"):
        evaluate(np.array([[0],[0]]),[{"easy":[],"hard":[1],"junk":[]}])

def test_normalize_and_atomic_feature_io(tmp_path):
    x=normalize([[0,0],[3,4]])
    np.testing.assert_allclose(x,[[0,0],[.6,.8]],atol=1e-6)
    path=tmp_path/"nested/features.npz"
    save_npz(path,features=x)
    with np.load(path) as z: np.testing.assert_array_equal(z["features"],x)
