"""Run lerobot-train with the "neuron" (AWS Trainium / TorchNeuron) device accepted.

lerobot hard-codes its device families (cuda | mps | xpu | cpu) in
lerobot.utils.device_utils and raises "Unknown device neuron" while building the policy
config — before any training code runs. TorchNeuron registers "neuron" as a normal
PyTorch device (PrivateUse1), so the only change needed is to teach those three helpers
the name. We patch them BEFORE lerobot's config modules import them (they bind the
names with `from ... import`), then hand off to lerobot's own entry point unchanged.

Used only when train.py resolves device=neuron:  python -m neuron_shim <lerobot args>
"""
import runpy
import sys

import torch
from lerobot.utils import device_utils as du

_orig_available = du.is_torch_device_available
_orig_safe = du.get_safe_torch_device
_orig_amp = du.is_amp_available


def _neuron_ok():
    mod = getattr(torch, "neuron", None)
    try:
        return mod is not None and mod.device_count() > 0
    except Exception:
        return False


def is_torch_device_available(try_device):
    if str(try_device).startswith("neuron"):
        return _neuron_ok()
    return _orig_available(try_device)


def get_safe_torch_device(try_device, log=False):
    if str(try_device).startswith("neuron"):
        if not _neuron_ok():
            raise ValueError(f"Requested device {try_device!r} but no Neuron device is visible.")
        return torch.device(str(try_device))
    return _orig_safe(try_device, log)


def is_amp_available(device):
    if str(device).startswith("neuron"):
        return False  # lerobot's AMP path assumes cuda/xpu/cpu autocast; bf16 comes from the model dtype
    return _orig_amp(device)


# safetensors' Rust loader only knows cpu/cuda/mps/...; "device neuron is invalid" while
# lerobot loads the pretrained weights. Load those to CPU instead — load_state_dict copies
# them into the model, and lerobot moves the policy to config.device afterwards.
import safetensors.torch as _st

_orig_load_file = _st.load_file


def _load_file(filename, device="cpu", *args, **kwargs):
    if str(device).startswith("neuron"):
        device = "cpu"
    return _orig_load_file(filename, device, *args, **kwargs)


_st.load_file = _load_file

# Under torchrun each rank must own one NeuronCore before accelerate/lerobot touch the device.
import os as _os

if "LOCAL_RANK" in _os.environ and _neuron_ok():
    torch.neuron.set_device(int(_os.environ["LOCAL_RANK"]))

du.is_torch_device_available = is_torch_device_available
du.get_safe_torch_device = get_safe_torch_device
du.is_amp_available = is_amp_available

if __name__ == "__main__":
    sys.argv = ["lerobot_train", *sys.argv[1:]]
    runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__", alter_sys=True)
