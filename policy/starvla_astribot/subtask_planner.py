from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_QWEN3_30B_PATH = os.environ.get(
    "ACTIVEARENA_PLANNER_VLM",
    os.environ.get(
        "STARVLA_PLANNER_VLM",  # deprecated compatibility alias
        str(
            Path(__file__).resolve().parents[2]
            / "../ActiveArena-VLA/playground/Pretrained_models/Qwen3-VL-30B-A3B-Instruct"
        ),
    ),
)

PROMPT_VERSION = "robotwin_subtask_equivalence_v1"

SYSTEM_PROMPT = """You are a robot subtask boundary judge.
Your job is to decide whether two subtask descriptions are the same executable robot subtask.
Use a strict slot match. They are the same only when ALL of these match:
1. primitive goal: search, pick/grasp, inspect/check, press, shake, place/put, stack, etc.
2. manipulated object and every distinguishing attribute such as color, size, side, or index
3. reference/destination object and spatial relation, if any
4. intended outcome

Treat wording changes and synonyms as the same when the manipulated object(s), reference object(s),
spatial relation, and goal state are the same. Examples of same wording:
- pick up / grasp / grab / take / hold
- press / push / tap / click
- place / put / drop when the destination and relation are unchanged
- find / locate / search for

Return different when any of these changes:
- manipulated object, reference object, destination object, spatial relation, or target attribute
- primitive goal, such as picking an object versus placing it somewhere
- same primitive repeated for a different object
- inspecting/checking an object versus picking/grasping that same object

Examples:
- "find apple and pick it up" vs "grasp apple" => {"same": true}
- "find apple and pick it up" vs "find banana and pick it up" => {"same": false}
- "find apple and pick it up" vs "find table and place apple on it" => {"same": false}
- "find large block and place medium block to its right" vs "find small block and pick it up" => {"same": false}
- "pick up gray target block" vs "inspect the pad color on gray target block" => {"same": false}
- "find medium block and grasp it" vs "find small block and grasp it" => {"same": false}

Return only one compact JSON object with these fields:
{"same": true|false, "confidence": 0.0-1.0, "normalized_previous": "...", "normalized_current": "...", "reason": "..."}
Keep reason under 12 words. If your reason mentions a changed object, goal, primitive, or relation, same must be false."""

COMPACT_SYSTEM_PROMPT = """You judge robot subtask boundaries.
Two texts are the same executable task only if the action, manipulated object and attributes,
destination or spatial relation, and intended outcome all match. Synonyms count as the same.
Return exactly one JSON object with one boolean field and no explanation:
{\"same\":true} or {\"same\":false}."""


@dataclass
class SubtaskDecision:
    same: bool
    confidence: float
    normalized_previous: str
    normalized_current: str
    reason: str
    raw_text: str = ""
    latency_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SubtaskPlannerError(RuntimeError):
    pass


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("\"' ")


def build_subtask_equivalence_messages(
    previous_subtask: str,
    current_subtask: str,
    *,
    task_instruction: Optional[str] = None,
    candidate_subtasks: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    candidate_lines = []
    for idx, item in enumerate(candidate_subtasks or [], start=1):
        text = _clean_text(item)
        if text:
            candidate_lines.append(f"{idx}. {text}")

    user_parts = []
    if task_instruction:
        user_parts.append(f"Full task instruction: {_clean_text(task_instruction)}")
    if candidate_lines:
        user_parts.append("Known valid subtasks for this task:\n" + "\n".join(candidate_lines))
    user_parts.extend(
        [
            f"Previous accepted subtask: {_clean_text(previous_subtask)}",
            f"Current candidate subtask: {_clean_text(current_subtask)}",
            "Are the previous and current subtasks the same executable subtask?",
        ]
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": "\n\n".join(user_parts)}]},
    ]


def build_subtask_equivalence_prompt(
    previous_subtask: str,
    current_subtask: str,
    *,
    task_instruction: Optional[str] = None,
    candidate_subtasks: Optional[Iterable[str]] = None,
) -> str:
    messages = build_subtask_equivalence_messages(
        previous_subtask,
        current_subtask,
        task_instruction=task_instruction,
        candidate_subtasks=candidate_subtasks,
    )
    return "\n\n".join(
        f"{message['role'].upper()}: {message['content'][0]['text']}" for message in messages
    )


def _plain_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    plain_messages = []
    for message in messages:
        parts = []
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                elif item:
                    parts.append(str(item))
            content_text = "\n".join(parts)
        else:
            content_text = str(content)
        plain_messages.append({"role": str(message.get("role", "user")), "content": content_text})
    return plain_messages


def _find_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        raise SubtaskPlannerError(f"planner response is not JSON: {text[:300]!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise SubtaskPlannerError(f"planner response JSON parse failed: {text[:300]!r}") from exc


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "same", "1"}:
        return True
    if text in {"false", "no", "different", "0"}:
        return False
    raise SubtaskPlannerError(f"planner returned invalid same value: {value!r}")


def parse_subtask_decision(raw_text: str, *, latency_sec: float = 0.0) -> SubtaskDecision:
    data = _find_json_object(raw_text)
    same_value = data.get("same", data.get("same_subtask", data.get("is_same", None)))
    if same_value is None:
        raise SubtaskPlannerError(f"planner JSON missing same field: {data}")
    confidence = data.get("confidence", 1.0)
    try:
        confidence_f = float(confidence)
    except Exception:
        confidence_f = 1.0
    confidence_f = max(0.0, min(1.0, confidence_f))
    same = _as_bool(same_value)
    reason = _clean_text(data.get("reason", ""))
    reason_l = reason.lower()
    contradiction_markers = (
        "different goal",
        "different goals",
        "different primitive",
        "different object",
        "different objects",
        "object differs",
        "objects differ",
        "attribute differs",
        "relation differs",
        "not the same",
        "mismatch",
    )
    if same and any(marker in reason_l for marker in contradiction_markers):
        same = False
        confidence_f = min(confidence_f, 0.9)
    return SubtaskDecision(
        same=same,
        confidence=confidence_f,
        normalized_previous=_clean_text(data.get("normalized_previous", "")),
        normalized_current=_clean_text(data.get("normalized_current", "")),
        reason=reason,
        raw_text=raw_text,
        latency_sec=float(latency_sec),
    )


def apply_candidate_distinct_guard(
    decision: SubtaskDecision,
    previous_subtask: str,
    current_subtask: str,
    candidate_subtasks: Optional[Iterable[str]] = None,
) -> SubtaskDecision:
    candidates = {_clean_text(item).lower() for item in (candidate_subtasks or []) if _clean_text(item)}
    previous_l = _clean_text(previous_subtask).lower()
    current_l = _clean_text(current_subtask).lower()
    if previous_l and current_l and previous_l != current_l and previous_l in candidates and current_l in candidates:
        decision.same = False
        decision.confidence = max(float(decision.confidence), 0.99)
        decision.reason = "distinct known candidate subtasks"
    return decision


def extract_subtask_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in (
            "subtask",
            "current_subtask",
            "subtask_instruction",
            "subtask_text",
            "vlm_text",
            "text",
            "output",
            "generated_text",
        ):
            if key in value:
                found = extract_subtask_text(value[key])
                if found:
                    return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = extract_subtask_text(item)
            if found:
                return found
        return None

    text = _clean_text(value)
    if not text:
        return None

    tag_match = re.search(r"<subtask>\s*(.*?)\s*</subtask>", text, flags=re.I | re.S)
    if tag_match:
        return _clean_text(tag_match.group(1))

    json_match = re.search(r"\{.*\}", text, flags=re.S)
    if json_match:
        try:
            found = extract_subtask_text(json.loads(json_match.group(0)))
            if found:
                return found
        except Exception:
            pass

    quoted_patterns = [
        r"Now\s+the\s+(?:task|subtask)\s+is\s+[\"']([^\"']+)[\"']",
        r"current\s+subtask\s*(?:is|:)\s*[\"']([^\"']+)[\"']",
        r"subtask\s*(?:is|:)\s*[\"']([^\"']+)[\"']",
    ]
    for pattern in quoted_patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return _clean_text(match.group(1))

    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.I | re.S)
    if think_match:
        think_text = _clean_text(think_match.group(1))
        for pattern in quoted_patterns:
            match = re.search(pattern, think_text, flags=re.I | re.S)
            if match:
                return _clean_text(match.group(1))
        return think_text if len(think_text) <= 180 else None

    if "<action>" in text.lower():
        return None
    return text if len(text) <= 180 else None


class HTTPSubtaskPlanner:
    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.url = str(url)
        self.timeout = float(timeout)

    def compare(
        self,
        previous_subtask: str,
        current_subtask: str,
        *,
        task_instruction: Optional[str] = None,
        candidate_subtasks: Optional[Iterable[str]] = None,
    ) -> SubtaskDecision:
        payload = {
            "prompt_version": PROMPT_VERSION,
            "task_instruction": task_instruction,
            "previous_subtask": previous_subtask,
            "current_subtask": current_subtask,
            "candidate_subtasks": list(candidate_subtasks or []),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SubtaskPlannerError(f"HTTP planner request failed: {exc}") from exc
        latency = time.perf_counter() - start
        if "decision" in data and isinstance(data["decision"], dict):
            raw = json.dumps(data["decision"], ensure_ascii=False)
        else:
            raw = data.get("raw_text") or data.get("text") or json.dumps(data, ensure_ascii=False)
        decision = parse_subtask_decision(raw, latency_sec=latency)
        decision = apply_candidate_distinct_guard(
            decision,
            previous_subtask,
            current_subtask,
            candidate_subtasks,
        )
        if decision.latency_sec == 0.0:
            decision.latency_sec = latency
        return decision


class OpenAICompatibleSubtaskPlanner:
    """Call an OpenAI-compatible text endpoint with only the two VLA texts."""

    def __init__(
        self,
        url: str,
        *,
        model_name: str,
        timeout: float = 30.0,
        max_new_tokens: int = 16,
        api_key: Optional[str] = None,
        disable_reasoning: bool = True,
    ) -> None:
        self.url = str(url)
        self.model_name = str(model_name)
        self.timeout = float(timeout)
        self.max_new_tokens = int(max_new_tokens)
        self.api_key = _clean_text(api_key) or None
        self.disable_reasoning = bool(disable_reasoning)

    def compare(
        self,
        previous_subtask: str,
        current_subtask: str,
        *,
        task_instruction: Optional[str] = None,
        candidate_subtasks: Optional[Iterable[str]] = None,
    ) -> SubtaskDecision:
        del task_instruction, candidate_subtasks
        messages = [
            {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Previous text: {_clean_text(previous_subtask)}\n"
                    f"Current text: {_clean_text(current_subtask)}\n"
                    "Same executable robot subtask?"
                ),
            },
        ]
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.disable_reasoning:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SubtaskPlannerError(
                f"OpenAI-compatible planner returned HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SubtaskPlannerError(f"OpenAI-compatible planner request failed: {exc}") from exc
        latency = time.perf_counter() - start

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SubtaskPlannerError(
                f"OpenAI-compatible planner response has no assistant message: {data}"
            ) from exc
        raw_text = message.get("content")
        if not raw_text:
            raw_text = message.get("reasoning") or message.get("reasoning_content")
        if not raw_text:
            raise SubtaskPlannerError(
                f"OpenAI-compatible planner returned an empty assistant message: {data}"
            )
        return parse_subtask_decision(str(raw_text), latency_sec=latency)


class TransformersQwenSubtaskPlanner:
    def __init__(
        self,
        model_path: str = DEFAULT_QWEN3_30B_PATH,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 192,
    ) -> None:
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(str(dtype).lower(), torch.bfloat16)

        self.model_path = str(Path(model_path).expanduser())
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        config = AutoConfig.from_pretrained(self.model_path)
        model_cls = (
            Qwen3VLMoeForConditionalGeneration
            if str(getattr(config, "model_type", "")).lower() == "qwen3_vl_moe"
            else Qwen3VLForConditionalGeneration
        )
        try:
            self.model = model_cls.from_pretrained(
                self.model_path,
                dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
        except TypeError:
            self.model = model_cls.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
        self.model.eval()
        self.max_new_tokens = int(max_new_tokens)

    def compare(
        self,
        previous_subtask: str,
        current_subtask: str,
        *,
        task_instruction: Optional[str] = None,
        candidate_subtasks: Optional[Iterable[str]] = None,
    ) -> SubtaskDecision:
        import torch

        messages = build_subtask_equivalence_messages(
            previous_subtask,
            current_subtask,
            task_instruction=task_instruction,
            candidate_subtasks=candidate_subtasks,
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        first_param = next(self.model.parameters())
        inputs = inputs.to(first_param.device)
        start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        latency = time.perf_counter() - start
        prompt_len = inputs["input_ids"].shape[1]
        text = self.processor.tokenizer.batch_decode(
            generated_ids[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        decision = parse_subtask_decision(text, latency_sec=latency)
        return apply_candidate_distinct_guard(
            decision,
            previous_subtask,
            current_subtask,
            candidate_subtasks,
        )


class VLLMSubtaskPlanner:
    def __init__(
        self,
        model_path: str = DEFAULT_QWEN3_30B_PATH,
        *,
        dtype: str = "bfloat16",
        max_new_tokens: int = 192,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
    ) -> None:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.model_path = str(Path(model_path).expanduser())
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.llm = LLM(
            model=self.model_path,
            dtype=str(dtype),
            tensor_parallel_size=int(tensor_parallel_size),
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len),
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=int(max_new_tokens),
        )

    def _prompt(
        self,
        previous_subtask: str,
        current_subtask: str,
        *,
        task_instruction: Optional[str] = None,
        candidate_subtasks: Optional[Iterable[str]] = None,
    ) -> str:
        messages = build_subtask_equivalence_messages(
            previous_subtask,
            current_subtask,
            task_instruction=task_instruction,
            candidate_subtasks=candidate_subtasks,
        )
        return self.processor.apply_chat_template(
            _plain_chat_messages(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

    def compare_many(self, requests: Iterable[dict[str, Any]]) -> list[Any]:
        request_list = list(requests)
        if not request_list:
            return []
        prompts = [
            self._prompt(
                item["previous_subtask"],
                item["current_subtask"],
                task_instruction=item.get("task_instruction"),
                candidate_subtasks=item.get("candidate_subtasks"),
            )
            for item in request_list
        ]
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        latency = time.perf_counter() - start
        per_item_latency = latency / max(len(outputs), 1)
        results = []
        for item, output in zip(request_list, outputs):
            try:
                text = output.outputs[0].text if output.outputs else ""
                decision = parse_subtask_decision(text, latency_sec=per_item_latency)
                results.append(
                    apply_candidate_distinct_guard(
                        decision,
                        item["previous_subtask"],
                        item["current_subtask"],
                        item.get("candidate_subtasks"),
                    )
                )
            except SubtaskPlannerError as exc:
                results.append(exc)
        return results

    def compare(
        self,
        previous_subtask: str,
        current_subtask: str,
        *,
        task_instruction: Optional[str] = None,
        candidate_subtasks: Optional[Iterable[str]] = None,
    ) -> SubtaskDecision:
        result = self.compare_many(
            [
                {
                    "previous_subtask": previous_subtask,
                    "current_subtask": current_subtask,
                    "task_instruction": task_instruction,
                    "candidate_subtasks": list(candidate_subtasks or []),
                }
            ]
        )[0]
        if isinstance(result, Exception):
            raise result
        return result


def build_subtask_planner(
    backend: str = "http",
    *,
    url: str = "http://127.0.0.1:7991/classify",
    model_path: str = DEFAULT_QWEN3_30B_PATH,
    timeout: float = 30.0,
    device_map: str = "auto",
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    max_new_tokens: int = 192,
    tensor_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    model_name: str = "qwen3.5-9b",
    api_key: Optional[str] = None,
    disable_reasoning: bool = True,
) -> Any:
    backend_l = str(backend or "http").lower()
    if backend_l in {"http", "server"}:
        return HTTPSubtaskPlanner(url=url, timeout=timeout)
    if backend_l in {"openai", "openai_compatible", "vllm_server"}:
        return OpenAICompatibleSubtaskPlanner(
            url=url,
            model_name=model_name,
            timeout=timeout,
            max_new_tokens=max_new_tokens,
            api_key=api_key,
            disable_reasoning=disable_reasoning,
        )
    if backend_l in {"transformers", "local", "qwen", "qwen3"}:
        return TransformersQwenSubtaskPlanner(
            model_path=model_path,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_implementation,
            max_new_tokens=max_new_tokens,
        )
    if backend_l in {"vllm", "vllm_local"}:
        return VLLMSubtaskPlanner(
            model_path=model_path,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    raise ValueError(f"Unsupported subtask planner backend: {backend!r}")
