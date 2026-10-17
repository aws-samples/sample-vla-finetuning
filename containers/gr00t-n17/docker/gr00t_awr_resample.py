#!/usr/bin/env python3
"""Advantage-weighted episode RESAMPLING for GR00T fine-tune (the AWR/RWR update-half).

WHY THIS EXISTS
  We want advantage-weighted regression (upweight high-advantage rollout episodes, so the
  fine-tune pulls the policy toward successful place behaviour and away from the 0/10 failure
  mode) WITHOUT patching any training internal. Per-*transition* AWR is not reachable in this
  stack: launch_finetune feeds a STREAMING IterableDataset (ShardedMixtureDataset) and the
  Gr00tN1d7DataCollator drops every non-modality key, so a per-sample advantage can never
  reach Gr00tTrainer.compute_loss (which returns an already-mean-reduced scalar). Verified vs
  NVIDIA/Isaac-GR00T @ 65cc4a192e6d (processing_gr00t_n1d7.py collator; trainer.py compute_loss;
  gr00t_n1d7.py flow-matching loss reduced to a scalar before it leaves the model).

  The faithful, robust equivalent is EPISODE-LEVEL advantage-weighted resampling (the RWR /
  reward-weighted-regression / filtered-BC family): materialise a dataset in which episode i
  appears with multiplicity m_i proportional to exp(advantage_i / beta), then run the
  BYTE-IDENTICAL verified launch_finetune (or g1_finetune.py) on it. Zero training-internal
  patching; the coarser episode-return weighting is actually well matched to our sparse
  success/place signal. HONEST LIMITATION: this weights whole episodes, not individual
  transitions — documented, and acceptable for the sparse-reward PoC.

WHAT IT DOES (all on the ALREADY-LOCAL LeRobot v2.1 dataset dir, before stats.py runs)
  1. Read per-episode advantages (JSON {"beta"?, "advantages": {"<episode_index>": adv}} or a
     bare {"<episode_index>": adv} map). S3 uri (--advantages-s3) is downloaded via boto3.
  2. compute_multiplicities(): w_i = exp((adv_i - amax)/beta); normalise so the least-favoured
     KEPT episode = 1 copy; round; clamp to [1, max_dup]. Upsample-only by default. With
     --allow-drop, episodes with adv < drop_below (default 0.0) get m=0 (filtered BC).
  3. For each episode needing >1 copies, PHYSICALLY duplicate it under a fresh episode_index:
     copy its parquet (rewriting the `episode_index` and global `index` columns) and its video
     file(s), append a meta/episodes.jsonl entry, and bump meta/info.json counters. Originals
     are never mutated (append-only) unless --allow-drop rewrites episodes.jsonl to exclude
     dropped episodes (orphan parquet/videos are harmless — the loader treats episodes.jsonl as
     the source of truth: len(dataset) == len(episodes_metadata), verified @65cc4a192e6d).

  UNSET GROOT_AWR_ADVANTAGES_S3 => the bootstrap never calls this => byte-identical to the
  existing GR00T fine-tune. So this is safe to bake into the shared gr00t-n17 image.

STACK: stdlib + boto3 (download) + pyarrow (parquet rewrite). All present in the gr00t venv
  (pyarrow is a LeRobot dependency in the locked closure). The math (compute_multiplicities,
  video-key extraction, path formatting) is pure and self-tested offline with NO GPU / NO
  network — the LeRobot loader's actual acceptance of the reshaped dataset is only confirmed by
  the first GPU run (a 1-shot bootstrap gap this offline gate cannot prove).
"""
import argparse
import json
import math
import os
import shutil
import sys


# LeRobot v2.1 info.json defaults (used only if the field is absent; real datasets carry them).
_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _load_advantages(path_or_uri, region=None):
    """Return (advantages: dict[int,float], beta_override: float|None). Accepts a local path or
    an s3:// uri (downloaded via boto3 to a temp file)."""
    if path_or_uri.startswith("s3://"):
        import boto3
        rest = path_or_uri[len("s3://"):]
        bucket, _, key = rest.partition("/")
        local = "/tmp/gr00t_awr_advantages.json"
        boto3.client("s3", region_name=region).download_file(bucket, key, local)
        path_or_uri = local
    with open(path_or_uri) as f:
        raw = json.load(f)
    beta = None
    if isinstance(raw, dict) and "advantages" in raw:
        beta = raw.get("beta")
        raw = raw["advantages"]
    if not isinstance(raw, dict):
        raise ValueError(f"advantages must be a map episode_index->advantage, got {type(raw)}")
    # keys may be str (JSON) or int; values must be numeric.
    adv = {int(k): float(v) for k, v in raw.items()}
    return adv, (float(beta) if beta is not None else None)


def compute_multiplicities(advantages, beta, max_dup, allow_drop=False, drop_below=0.0):
    """Pure AWR math. advantages: dict[int,float]. Returns dict[int,int] multiplicity (total
    copies incl. the original; 0 == drop). Upsample-only unless allow_drop.

    w_i = exp((adv_i - amax)/beta)  (amax subtracted for numerical stability; the max-advantage
    episode has w=1). Normalise by the smallest KEPT weight so the least-favoured kept episode
    maps to 1 copy; round; clamp to [1, max_dup]. Independent of files -> unit-testable.
    """
    if not advantages:
        return {}
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    kept = {ep: a for ep, a in advantages.items()
            if not (allow_drop and a < drop_below)}
    mult = {ep: 0 for ep in advantages}  # dropped episodes stay 0
    if not kept:
        return mult  # degenerate (everything dropped) — caller guards against empty dataset
    amax = max(kept.values())
    weights = {ep: math.exp((a - amax) / beta) for ep, a in kept.items()}
    wmin = min(weights.values())
    for ep, w in weights.items():
        m = int(round(w / wmin))  # >= 1 by construction (wmin is the smallest)
        m = max(1, min(max_dup, m))
        mult[ep] = m
    return mult


def _video_keys(info):
    """Feature keys whose dtype is 'video' (each needs its own per-episode .mp4 copied)."""
    feats = info.get("features", {})
    return [k for k, spec in feats.items()
            if isinstance(spec, dict) and spec.get("dtype") == "video"]


def _rewrite_columns(table, new_episode_index, global_start):
    """Return a pyarrow table with `episode_index` set to new_episode_index and the global
    `index` column set to a fresh contiguous range [global_start, global_start+len). Columns
    absent from the parquet are left untouched (defensive vs schema variation)."""
    import pyarrow as pa
    n = table.num_rows
    for name, values in (
        ("episode_index", [new_episode_index] * n),
        ("index", list(range(global_start, global_start + n))),
    ):
        if name in table.column_names:
            idx = table.column_names.index(name)
            arr = pa.array(values, type=table.schema.field(name).type)
            table = table.set_column(idx, table.schema.field(name), arr)
    return table


def resample_dataset(data_dir, advantages, beta, max_dup, allow_drop=False, drop_below=0.0):
    """Apply advantage-weighted episode resampling in place on the LeRobot v2.1 dataset at
    data_dir. Returns a summary dict. Existing files are never mutated; new copies are added
    (and episodes.jsonl is rewritten only when episodes are dropped)."""
    import pyarrow.parquet as pq

    meta_dir = os.path.join(data_dir, "meta")
    info_path = os.path.join(meta_dir, "info.json")
    episodes_path = os.path.join(meta_dir, "episodes.jsonl")
    with open(info_path) as f:
        info = json.load(f)
    with open(episodes_path) as f:
        episodes = [json.loads(line) for line in f if line.strip()]

    chunk_size = int(info.get("chunks_size", _DEFAULT_CHUNK_SIZE))
    data_pat = info.get("data_path", _DEFAULT_DATA_PATH)
    video_pat = info.get("video_path", _DEFAULT_VIDEO_PATH)
    vkeys = _video_keys(info)

    mult = compute_multiplicities(advantages, beta, max_dup, allow_drop, drop_below)

    existing_ep_ids = [int(e["episode_index"]) for e in episodes]
    len_by_id = {int(e["episode_index"]): int(e["length"]) for e in episodes}
    orig_total_frames = int(info.get("total_frames", sum(len_by_id.values())))

    next_ep = max(existing_ep_ids) + 1 if existing_ep_ids else 0
    next_global = orig_total_frames  # new copies get global indices after all originals

    dropped = [e for e in existing_ep_ids if mult.get(e, 1) == 0]
    added_meta = []  # episodes.jsonl entries for the new copies
    n_copies = 0

    for ep in episodes:
        ep_id = int(ep["episode_index"])
        m = mult.get(ep_id, 1)  # episodes without an advantage stay at 1 copy
        if m <= 1:
            continue  # original already present (m==1) or dropped (m==0, handled below)
        for _ in range(m - 1):
            _duplicate_episode(data_dir, ep, ep_id, next_ep, next_global,
                               chunk_size, data_pat, video_pat, vkeys)
            copy_meta = dict(ep)
            copy_meta["episode_index"] = next_ep
            added_meta.append(copy_meta)
            next_global += len_by_id[ep_id]
            next_ep += 1
            n_copies += 1

    # Rewrite episodes.jsonl: drop excluded originals (if any) then append the copies. When
    # nothing is dropped this is append-only (existing lines byte-preserved).
    kept = [e for e in episodes if int(e["episode_index"]) not in set(dropped)]
    new_episodes = kept + added_meta
    with open(episodes_path, "w") as f:
        for e in new_episodes:
            f.write(json.dumps(e) + "\n")

    # info.json counters: recompute over the final episode set.
    total_episodes = len(new_episodes)
    total_frames = sum(int(e["length"]) for e in new_episodes)
    max_idx = max((int(e["episode_index"]) for e in new_episodes), default=-1)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_chunks"] = (max_idx // chunk_size) + 1 if max_idx >= 0 else 0
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    summary = {
        "orig_episodes": len(episodes),
        "dropped": len(dropped),
        "copies_added": n_copies,
        "final_episodes": total_episodes,
        "final_frames": total_frames,
    }
    print(f"[gr00t-awr] resample summary: {json.dumps(summary)}", flush=True)
    return summary


def _duplicate_episode(data_dir, ep_meta, src_ep, dst_ep, global_start,
                       chunk_size, data_pat, video_pat, vkeys):
    """Copy episode src_ep's parquet (rewriting episode_index/index cols) and video file(s) to
    a new episode_index dst_ep. Creates chunk dirs as needed."""
    import pyarrow.parquet as pq

    src_chunk = src_ep // chunk_size
    dst_chunk = dst_ep // chunk_size

    # Parquet: read, rewrite the two index columns, write to the new path.
    src_parquet = os.path.join(data_dir, data_pat.format(episode_chunk=src_chunk,
                                                          episode_index=src_ep))
    dst_parquet = os.path.join(data_dir, data_pat.format(episode_chunk=dst_chunk,
                                                         episode_index=dst_ep))
    os.makedirs(os.path.dirname(dst_parquet), exist_ok=True)
    table = pq.read_table(src_parquet)
    table = _rewrite_columns(table, dst_ep, global_start)
    pq.write_table(table, dst_parquet)

    # Videos: byte copy per video key (no rewrite needed — mp4 carries no episode index).
    for vk in vkeys:
        src_vid = os.path.join(data_dir, video_pat.format(episode_chunk=src_chunk,
                                                          video_key=vk, episode_index=src_ep))
        dst_vid = os.path.join(data_dir, video_pat.format(episode_chunk=dst_chunk,
                                                         video_key=vk, episode_index=dst_ep))
        if os.path.exists(src_vid):
            os.makedirs(os.path.dirname(dst_vid), exist_ok=True)
            shutil.copy2(src_vid, dst_vid)


def main():
    ap = argparse.ArgumentParser(description="Advantage-weighted episode resampling (AWR/RWR).")
    ap.add_argument("--data-dir", required=True, help="local LeRobot v2.1 dataset root")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--advantages", help="local advantages JSON path")
    src.add_argument("--advantages-s3", help="s3:// uri to advantages JSON")
    ap.add_argument("--beta", type=float, default=None,
                    help="AWR temperature (env GROOT_AWR_BETA; JSON beta; default 1.0)")
    ap.add_argument("--max-dup", type=int,
                    default=int(os.environ.get("GROOT_AWR_MAX_DUP", "8")),
                    help="cap on copies per episode")
    ap.add_argument("--allow-drop", action="store_true",
                    default=bool(os.environ.get("GROOT_AWR_ALLOW_DROP")),
                    help="drop episodes with advantage < --drop-below (filtered BC)")
    ap.add_argument("--drop-below", type=float,
                    default=float(os.environ.get("GROOT_AWR_DROP_BELOW", "0.0")),
                    help="advantage threshold below which episodes are dropped")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION"))
    args = ap.parse_args()

    adv_path = args.advantages or args.advantages_s3
    advantages, beta_json = _load_advantages(adv_path, region=args.region)
    # beta precedence: CLI > env > JSON > 1.0.
    beta = args.beta
    if beta is None and os.environ.get("GROOT_AWR_BETA"):
        beta = float(os.environ["GROOT_AWR_BETA"])
    if beta is None:
        beta = beta_json if beta_json is not None else 1.0

    if not advantages:
        print("[gr00t-awr] no advantages provided — nothing to resample.", flush=True)
        return 0

    print(f"[gr00t-awr] resampling {len(advantages)} scored episodes "
          f"(beta={beta} max_dup={args.max_dup} allow_drop={args.allow_drop} "
          f"drop_below={args.drop_below})", flush=True)
    resample_dataset(args.data_dir, advantages, beta, args.max_dup,
                     allow_drop=args.allow_drop, drop_below=args.drop_below)
    return 0


if __name__ == "__main__":
    sys.exit(main())
