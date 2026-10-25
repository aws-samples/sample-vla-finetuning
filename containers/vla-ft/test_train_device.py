"""Unit test for train.resolve_device / build_command device wiring (no GPU, no AWS).

Guards the TorchNeuron (Trainium) opt-in: the default must stay the verified cuda
path, `neuron` must reach lerobot's --policy.device, a typo must fail loudly, and
neuron multi-core runs as torchrun data parallel (verified on trn2.48xlarge: 8 and 64
NeuronCores), and multi-node neuron is refused.
"""
import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location(
    "train", os.path.join(os.path.dirname(__file__), "src", "train.py")
)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

PASS, FAIL = 0, 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


for k in ("NNODES", "NODE_RANK", "MASTER_ADDR", "SM_NUM_GPUS"):
    os.environ.pop(k, None)

check(train.resolve_device({}) == "cuda", "default device = cuda (verified path unchanged)")
for d in ("cuda", "neuron", "cpu"):
    check(train.resolve_device({"device": d}) == d, f"explicit device={d}")
check(train.resolve_device({"device": " Neuron "}) == "neuron", "device is trimmed + lowercased")
check(raises(lambda: train.resolve_device({"device": "nueron"})), "typo raises (never silently CPU)")
check(train.resolve_device({"device": "auto"}) in ("cuda", "cpu", "neuron"), "auto resolves to a known device")

cmd = train.build_command({"policy": "smolvla", "device": "neuron"})
check("--policy.device=neuron" in cmd, "neuron reaches --policy.device")
check("--policy.device=cuda" not in cmd, "no stray cuda flag when neuron")
check(not any(c == "accelerate" for c in cmd), "neuron uses the single-process launcher")
check(cmd[1:3] == ["-m", "neuron_shim"], "neuron runs lerobot through neuron_shim")
check("--device=neuron" not in cmd, "device is consumed, not passed through as a bare lerobot flag")
cmd = train.build_command({"policy": "smolvla"})
check(cmd[1:3] == ["-m", "lerobot.scripts.lerobot_train"], "cuda path still runs lerobot directly")
check("--policy.device=cuda" in cmd, "default command keeps --policy.device=cuda")
cmd = train.build_command({"policy": "smolvla", "device": "neuron", "num_gpus": 8})
check(cmd[:2] == ["torchrun", "--standalone"], "neuron + num_gpus>1 uses torchrun (one process per NeuronCore)")
check("--nproc_per_node=8" in cmd, "num_gpus becomes nproc_per_node")
check(cmd[cmd.index("-m") + 1] == "neuron_shim", "neuron data parallel still goes through neuron_shim")
check(not any(c == "accelerate" for c in cmd), "neuron data parallel does not use accelerate launch")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
