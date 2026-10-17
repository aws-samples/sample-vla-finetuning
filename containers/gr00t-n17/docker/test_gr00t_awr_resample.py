#!/usr/bin/env python3
"""Offline self-test for the AWR/RWR episode resampler (gr00t_awr_resample.py).

Runs with NO GPU / NO network / NO gr00t stack — only pyarrow (a LeRobot dep, present in the
image venv and locally). It exercises the two pieces of genuinely new logic:
  1. compute_multiplicities() — the pure AWR math (monotonicity, beta flattening, drop, clamp).
  2. resample_dataset() — the LeRobot v2.1 file surgery on a synthetic dataset (episode_index /
     global index column rewrite, per-episode video copy, meta/episodes.jsonl + info.json).

NOTE (1-shot gap): this proves the reshaped dataset is SELF-CONSISTENT, not that the pinned
GR00T loader ACCEPTS it — that is a boot-time gap only the first GPU run confirms. See
gr00t_awr_resample.py header and the session note.
"""
import importlib.util
import json
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "awr", os.path.join(_HERE, "gr00t_awr_resample.py"))
awr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(awr)

CHUNK = 1000
DATA_PAT = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PAT = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


# ── 1. pure AWR math ────────────────────────────────────────────────────────────────
def test_multiplicities_monotone():
    m = awr.compute_multiplicities({0: 2.0, 1: 1.0, 2: 0.0}, beta=1.0, max_dup=8)
    assert m[0] >= m[1] >= m[2] >= 1  # higher advantage -> at least as many copies


def test_multiplicities_high_beta_flattens():
    m = awr.compute_multiplicities({0: 2.0, 1: 1.0, 2: 0.0}, beta=100.0, max_dup=8)
    assert set(m.values()) == {1}  # temperature -> uniform


def test_multiplicities_drop_below():
    m = awr.compute_multiplicities({0: 2.0, 1: 1.0, 2: 0.0}, beta=1.0, max_dup=8,
                                   allow_drop=True, drop_below=0.5)
    assert m[2] == 0 and m[0] >= 1 and m[1] >= 1


def test_multiplicities_max_dup_clamp():
    m = awr.compute_multiplicities({0: 5.0, 1: 0.0}, beta=0.3, max_dup=2)
    assert max(m.values()) <= 2


def test_multiplicities_empty():
    assert awr.compute_multiplicities({}, beta=1.0, max_dup=8) == {}


# ── 2. LeRobot v2.1 file surgery ──────────────────────────────────────────────────────
def _make_dataset(d, lens):
    os.makedirs(os.path.join(d, "meta"))
    gstart = 0
    for ep, L in lens.items():
        ch = ep // CHUNK
        p = os.path.join(d, DATA_PAT.format(episode_chunk=ch, episode_index=ep))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        pq.write_table(pa.table({
            "episode_index": pa.array([ep] * L, pa.int64()),
            "index": pa.array(list(range(gstart, gstart + L)), pa.int64()),
            "action": pa.array([[0.1, 0.2]] * L),  # modality col: must survive untouched
        }), p)
        vp = os.path.join(d, VIDEO_PAT.format(episode_chunk=ch,
                                              video_key="observation.images.cam",
                                              episode_index=ep))
        os.makedirs(os.path.dirname(vp), exist_ok=True)
        with open(vp, "wb") as f:
            f.write(b"FAKEMP4:%d" % ep)
        gstart += L
    with open(os.path.join(d, "meta", "episodes.jsonl"), "w") as f:
        for ep, L in lens.items():
            f.write(json.dumps({"episode_index": ep, "length": L, "tasks": ["pick apple"]}) + "\n")
    with open(os.path.join(d, "meta", "info.json"), "w") as f:
        json.dump({
            "total_episodes": len(lens), "total_frames": sum(lens.values()),
            "total_chunks": 1, "chunks_size": CHUNK,
            "data_path": DATA_PAT, "video_path": VIDEO_PAT,
            "features": {"observation.images.cam": {"dtype": "video"},
                         "action": {"dtype": "float32"}},
        }, f)


def test_resample_upsample_and_drop():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, {0: 4, 1: 3, 2: 2})
        # ep0 best, ep2 dropped (adv<0.5). mult(beta=1,drop): {0:3, 1:1, 2:0}.
        summary = awr.resample_dataset(d, {0: 2.0, 1: 1.0, 2: 0.0}, beta=1.0, max_dup=8,
                                       allow_drop=True, drop_below=0.5)
        eps = [json.loads(l) for l in open(os.path.join(d, "meta", "episodes.jsonl")) if l.strip()]
        info = json.load(open(os.path.join(d, "meta", "info.json")))
        ep_ids = [e["episode_index"] for e in eps]

        # kept ep0,ep1 (+2 new copies of ep0), ep2 dropped -> ids {0,1,3,4}
        assert set(ep_ids) == {0, 1, 3, 4}, ep_ids
        assert len(set(ep_ids)) == len(ep_ids)              # unique episode_index
        assert info["total_episodes"] == 4
        assert info["total_frames"] == 4 + 3 + 2 * 4        # ep0(4)+ep1(3)+2*ep0(4)=15
        assert summary["copies_added"] == 2 and summary["dropped"] == 1

        # each parquet: episode_index col rewritten to its own id; modality col survives; global
        # index unique across the whole reshaped dataset (no collisions).
        all_global = []
        for e in eps:
            ep = e["episode_index"]; ch = ep // CHUNK
            tt = pq.read_table(os.path.join(d, DATA_PAT.format(episode_chunk=ch, episode_index=ep)))
            assert set(tt.column("episode_index").to_pylist()) == {ep}
            assert "action" in tt.column_names
            all_global += tt.column("index").to_pylist()
            vp = os.path.join(d, VIDEO_PAT.format(episode_chunk=ch,
                                                  video_key="observation.images.cam",
                                                  episode_index=ep))
            assert os.path.exists(vp)                        # video copied per new episode
        assert len(all_global) == len(set(all_global)) == 15  # no global index collision


def test_resample_no_advantages_is_noop_on_meta():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, {0: 4, 1: 3})
        before = open(os.path.join(d, "meta", "episodes.jsonl")).read()
        # all equal advantage, no drop -> multiplicities all 1 -> no copies added.
        awr.resample_dataset(d, {0: 1.0, 1: 1.0}, beta=1.0, max_dup=8)
        after = open(os.path.join(d, "meta", "episodes.jsonl")).read()
        assert before == after  # append-only path with nothing to append == byte-identical
        assert json.load(open(os.path.join(d, "meta", "info.json")))["total_episodes"] == 2


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok: {fn.__name__}")
    print(f"ALL {len(fns)} AWR RESAMPLE TESTS PASSED")
    sys.exit(0)
