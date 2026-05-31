import json
import os
import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from base_captioner import VideoCaptioner

CHAI_MODEL = os.getenv("CHAI_MODEL", "chancharikm/CHAI_SFT_model_8b")

_CAPTION_PROMPT = (
    "Analyze this video and return a JSON object with exactly these fields:\n"
    '- "caption": one sentence describing the overall video\n'
    '- "scene": the setting or environment\n'
    '- "events": list of objects, each with "start" (seconds), "end" (seconds), "description"\n'
    "Return only the JSON object, no other text."
)

_FIND_PROMPT = (
    'Find when "{event}" occurs in this video.\n'
    'Return JSON: {{"raw": "<description>", "start": <float or null>, "end": <float or null>}}\n'
    "Return only the JSON, no other text."
)


class ChaiCaptioner(VideoCaptioner):
    def load_model(self) -> None:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            CHAI_MODEL,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(CHAI_MODEL)

    def _infer(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        video_kwargs = {k: v[0] if isinstance(v, list) else v for k, v in video_kwargs.items()}
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self._model.device)
        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def caption(self, path: str) -> dict:
        raw = self._infer([{"role": "user", "content": [
            {"type": "video", "video": path},
            {"type": "text", "text": _CAPTION_PROMPT},
        ]}], max_new_tokens=512)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            data = json.loads(m.group()) if m else {}
            return {
                "caption": data.get("caption", raw),
                "scene": data.get("scene", ""),
                "events": data.get("events", []),
            }
        except json.JSONDecodeError:
            return {"caption": raw, "scene": "", "events": []}

    def caption_with_prompt(self, path: str, prompt: str, max_new_tokens: int = 512) -> str:
        return self._infer([{"role": "user", "content": [
            {"type": "video", "video": path},
            {"type": "text", "text": prompt},
        ]}], max_new_tokens=max_new_tokens)

    def find(self, path: str, event: str) -> dict:
        raw = self._infer([{"role": "user", "content": [
            {"type": "video", "video": path},
            {"type": "text", "text": _FIND_PROMPT.format(event=event)},
        ]}], max_new_tokens=128)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            data = json.loads(m.group()) if m else {}
            start, end = data.get("start"), data.get("end")
            span = (float(start), float(end)) if start is not None and end is not None else None
            return {"raw": data.get("raw", raw), "span": span, "format_ok": span is not None}
        except (json.JSONDecodeError, ValueError):
            return {"raw": raw, "span": None, "format_ok": False}
