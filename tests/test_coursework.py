"""P1-P6: CPU-only contract tests. Initially fail with NotImplementedError."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from PIL import Image
from .adapters import (
    run_protocol, run_search, run_pool, run_sift, run_rootsift,
    run_sample, run_vocabulary, run_vlad, run_inliers, run_rerank,
)
from instance_retrieval.evaluation import evaluate
pytestmark = pytest.mark.coursework

def test_protocol_sets_and_no_mutation():
    gnd = [{"easy": [1,2], "hard": [3], "junk": [0]}]
    original = copy.deepcopy(gnd)
    medium, hard = run_protocol(gnd,"medium")[0], run_protocol(gnd,"hard")[0]
    assert set(medium["ok"]) == {1,2,3} and set(medium["junk"]) == {0}
    assert set(hard["ok"]) == {3} and set(hard["junk"]) == {0,1,2}
    assert gnd == original

def test_protocol_invalid_name():
    with pytest.raises(ValueError):
        run_protocol([], "easy")

def test_protocol_official_metrics():
    result = evaluate(np.array([[0],[1],[2],[3]]),
                      [{"easy":[1],"hard":[3],"junk":[0]}])
    assert result["medium"]["mAP"] == pytest.approx(100*19/24)
    assert result["hard"]["mAP"] == pytest.approx(25)
    assert result["hard"]["mP@10"] == pytest.approx(50)

def test_protocol_empty_positive_query():
    gnd=[{"easy":[],"hard":[],"junk":[]},{"easy":[],"hard":[0],"junk":[]}]
    result=evaluate(np.array([[0,0],[1,1]]),gnd)
    assert result["hard"]["AP"] == [None,100.]
    assert result["hard"]["valid_queries"] == 1

def test_search_scores_orientation_and_stable_ties():
    db=np.array([[2.,0],[0,5],[1,0]])
    before=db.copy()
    ranks,scores=run_search(db,np.array([[7.,0],[0,3]]))
    np.testing.assert_array_equal(ranks,[[0,1],[2,0],[1,2]])
    np.testing.assert_allclose(scores,[[1,0],[0,1],[1,0]],atol=1e-6)
    assert ranks.shape == scores.shape == (3,2)
    assert np.issubdtype(ranks.dtype,np.integer)
    np.testing.assert_array_equal(db,before)

def test_search_zero_vectors():
    ranks,scores=run_search(np.zeros((3,4)),np.zeros((2,4)))
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(ranks,[[0,0],[1,1],[2,2]])

def test_pool_cls():
    tokens=torch.tensor([[[9.,-7.],[1.,2.],[3.,4.]]])
    result=run_pool(tokens,"cls")
    torch.testing.assert_close(result,torch.tensor([[9.,-7.]]))
    assert result.device == tokens.device and result.dtype == tokens.dtype

def test_pool_gem_excludes_cls():
    tokens=torch.tensor([[[100.,100.],[1.,2.],[3.,4.]]])
    torch.testing.assert_close(run_pool(tokens,"gem",p=2),torch.tensor([[5**.5,10**.5]]))

def test_pool_gem_negative_activations():
    tokens=torch.tensor([[[50.,50.],[-3.,4.],[2.,0.]]])
    torch.testing.assert_close(run_pool(tokens,"gem",p=1),torch.tensor([[1.,2.]]),atol=1e-5,rtol=0)

def test_pool_invalid_parameters():
    with pytest.raises(ValueError):
        run_pool(torch.ones(1,3,2),"unknown")
    with pytest.raises(ValueError):
        run_pool(torch.ones(1,3,2),"gem",p=0)

def test_sift_blank_image():
    xy,desc=run_sift(Image.new("RGB",(64,64)))
    assert xy.shape == (0,2) and desc.shape == (0,128)
    assert xy.dtype == np.float32 and desc.dtype == np.uint8

def test_sift_response_cap_and_storage(monkeypatch):
    import instance_retrieval.local as local
    points=[SimpleNamespace(response=r,pt=(i,i+1)) for i,r in enumerate([.2,.9,.5])]
    class Detector:
        def detectAndCompute(self,*args):
            return points,np.array([[1.2]*128,[300.]*128,[8.6]*128],np.float32)
    monkeypatch.setattr(local.cv2,"SIFT_create",lambda *a,**k: Detector())
    xy,desc=run_sift(Image.new("RGB",(30,30)),max_keypoints=2)
    np.testing.assert_array_equal(xy,[[1,2],[2,3]])
    assert desc.dtype == np.uint8 and desc.shape == (2,128)
    assert (desc[0] == 255).all() and (desc[1] == 9).all()

def test_rootsift_values_and_empty_rows():
    raw=np.zeros((2,128),np.uint8)
    raw[0,:2]=[1,3]
    original=raw.copy()
    out=run_rootsift(raw)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0,:2],[.5,3**.5/2],atol=1e-6)
    assert np.isfinite(out).all() and not out[1].any()
    np.testing.assert_array_equal(raw,original)
    assert run_rootsift(raw[:0]).shape == (0,128)

class TinyStore:
    def __init__(self):
        self.dataset=SimpleNamespace(imlist=["a","b"])
        self.desc=np.zeros((12,128),np.uint8)
        self.desc[np.arange(12),np.arange(12)]=1
    def get(self,index,query=False):
        assert not query,"Do not sample queries"
        block=self.desc[:5] if index == 0 else self.desc[5:]
        return np.zeros((len(block),2),np.float32),block

def test_sample_budget_seed_and_no_replacement():
    store=TinyStore()
    a=run_sample([store],count=7,seed=42)
    b=run_sample([store],count=7,seed=42)
    assert a.shape == (7,128) and a.dtype == np.float32
    np.testing.assert_array_equal(a,b)
    assert len(np.unique(a.argmax(1))) == 7
    assert run_sample([store],count=20).shape == (12,128)

def test_vocabulary_separated_clusters():
    rng=np.random.default_rng(4)
    sample=np.r_[rng.normal(0,.001,(80,128)),rng.normal(1,.001,(80,128))].astype(np.float32)
    centers=run_vocabulary(sample,k=2,iterations=10)
    assert centers.shape == (2,128) and centers.dtype == np.float32
    np.testing.assert_allclose(np.sort(centers.mean(1)),[0,1],atol=.01)

def test_vocabulary_insufficient_sample():
    with pytest.raises(ValueError):
        run_vocabulary(np.zeros((1,128),np.float32),k=2)

def test_vlad_assignment_and_normalization():
    desc=np.zeros((2,128),np.uint8)
    desc[0,0],desc[1,1]=1,1
    centers=np.zeros((2,128),np.float32)
    centers[0,0],centers[1,1]=.8,.8
    expected=np.zeros(256,np.float32)
    expected[0]=expected[129]=2**-.5
    vector=run_vlad(desc,centers)
    np.testing.assert_allclose(vector,expected,atol=1e-6)
    np.testing.assert_allclose(run_vlad(np.repeat(desc,2,axis=0),centers),vector,atol=1e-6)

def test_vlad_empty_image():
    empty=run_vlad(np.empty((0,128),np.uint8),np.zeros((2,128),np.float32))
    assert empty.shape == (256,) and not empty.any() and np.isfinite(empty).all()

def test_geometry_known_planar_transform():
    rng=np.random.default_rng(123)
    xy=rng.uniform(0,200,(20,2)).astype(np.float32)
    desc=rng.integers(0,200,(20,128),dtype=np.uint8)
    assert run_inliers((xy,desc),(xy*1.1+10,desc)) == 20

def test_geometry_too_few_matches():
    xy=np.zeros((3,2),np.float32)
    desc=np.zeros((3,128),np.uint8)
    assert run_inliers((xy,desc),(xy,desc)) == 0

def test_rerank_stable_ties_tail_and_no_mutation(monkeypatch):
    import instance_retrieval.geometry as geometry
    class Store:
        def get(self,i,query=False): return ("q" if query else "d",i)
    counts={2:1,0:3,4:3,1:999,3:999}
    monkeypatch.setattr(geometry,"geometric_inliers",lambda q,d: counts[d[1]])
    ranks=np.array([[2],[0],[4],[1],[3]],np.int32)
    original=ranks.copy()
    result,inliers,elapsed=run_rerank(ranks,Store(),top_n=3)
    np.testing.assert_array_equal(result[:,0],[0,4,2,1,3])
    np.testing.assert_array_equal(inliers[:,0],[3,3,1])
    np.testing.assert_array_equal(ranks,original)
    assert elapsed >= 0

def test_rerank_invalid_top_n():
    with pytest.raises(ValueError):
        run_rerank(np.array([[0]]),None,top_n=0)
