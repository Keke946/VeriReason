"""Independent, local-only adapter hydration; base remains meta, no inference claim."""

from __future__ import annotations

import sys


def main():
    import json
    from pathlib import Path

    import torch
    from peft import PeftConfig, get_peft_model, get_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from verireason.m3_runtime import artifact_manifest

    path, config_path, expected = sys.argv[1:]
    if artifact_manifest(path)["manifest_sha256"] != expected:
        return 2
    values = json.loads(Path(config_path).read_text())
    model_type = values.pop("model_type")
    config = AutoConfig.for_model(model_type, **values)
    with torch.device("meta"):
        base = AutoModelForCausalLM.from_config(config)
    adapter = get_peft_model(base, PeftConfig.from_pretrained(path, local_files_only=True),
                            low_cpu_mem_usage=True)
    # Materialize ONLY LoRA modules on CPU; base weights remain meta. PEFT would
    # otherwise move a hydrated adapter back to its meta base layer's device.
    for name, module in adapter.named_modules():
        if name.endswith((".lora_A.default", ".lora_B.default")):
            module.to_empty(device="cpu", recurse=False)
    adapter.load_adapter(path, adapter_name="default", torch_device="cpu",
                         low_cpu_mem_usage=False, local_files_only=True)
    loaded = get_peft_model_state_dict(adapter)
    saved = load_peft_weights(path, device="cpu")
    if not loaded or loaded.keys() != saved.keys():
        return 3
    for key, value in loaded.items():
        if (value.is_meta or not torch.isfinite(value).all()
                or not torch.equal(value.cpu(), saved[key])):
            return 4
    # This verifies exact adapter hydration into declared module shapes, not a forward pass.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
