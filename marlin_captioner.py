import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from base_captioner import VideoCaptioner

MARLIN_MODEL = os.getenv("MARLIN_MODEL", "NemoStation/Marlin-2B")

# Marlin emits a <think>...</think> block at the start of raw generate() output.
_THINK_RE = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)


class MarlinCaptioner(VideoCaptioner):
    def load_model(self) -> None:
        self._model = AutoModelForCausalLM.from_pretrained(
            MARLIN_MODEL,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
        )
        self._processor = AutoProcessor.from_pretrained(MARLIN_MODEL, trust_remote_code=True)

    def caption(self, path: str) -> dict:
        # Marlin's custom modeling code returns {caption, scene, events} directly.
        return self._model.caption(path)

    def caption_with_prompt(self, path: str, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": path},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        return _THINK_RE.sub("", text).strip()

    def find(self, path: str, event: str) -> dict:
        # Marlin's custom modeling code returns {raw, span, format_ok} directly.
        return self._model.find(path, event)
