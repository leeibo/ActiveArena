import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CURRENT_FILE_PATH = Path(__file__).resolve()
DESCRIPTION_UTILS_DIR = CURRENT_FILE_PATH.parent
DESCRIPTION_ROOT = DESCRIPTION_UTILS_DIR.parent
OBJECT_DESCRIPTION_DIR = DESCRIPTION_ROOT / "objects_description"
TASK_INSTRUCTION_DIR = DESCRIPTION_ROOT / "task_instruction"


def extract_placeholders(instruction: str) -> List[str]:
    """Extract placeholders like {A} from an instruction string."""
    return re.findall(r"{([^}]+)}", str(instruction))


def normalize_episode_params(episode_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(episode_params, dict):
        return {}
    return {str(key).strip("{}"): value for key, value in episode_params.items()}


def _shuffle_copy(items: Iterable[str], rng=None) -> List[str]:
    shuffled = list(items)
    chooser = rng if rng is not None else random
    chooser.shuffle(shuffled)
    return shuffled


def _choice(items: List[str], rng=None) -> str:
    chooser = rng if rng is not None else random
    return chooser.choice(items)


def filter_instructions(
    instructions: Iterable[str],
    episode_params: Optional[Dict[str, Any]],
    rng=None,
    shuffle: bool = True,
) -> List[str]:
    """
    Keep only instructions whose placeholders match the available episode params.

    This mirrors the original episode-instruction pipeline, including the special
    handling where arm placeholders may be omitted from the instruction template.
    """
    stripped_episode_params = normalize_episode_params(episode_params)
    candidates = list(instructions)
    if shuffle:
        candidates = _shuffle_copy(candidates, rng=rng)

    arm_params = {
        key for key in stripped_episode_params.keys()
        if len(key) == 1 and "a" <= key <= "z"
    }

    filtered = []
    available_placeholders = set(stripped_episode_params.keys())
    for instruction in candidates:
        placeholders = set(extract_placeholders(instruction))
        # A template is valid as long as every placeholder it mentions can be
        # resolved for this episode.  Many natural instructions intentionally
        # omit some object/arm placeholders (for example "inspect the blocks"
        # may not mention the pads explicitly), so requiring exact equality
        # with all episode params incorrectly drops otherwise valid templates.
        if placeholders.issubset(available_placeholders):
            filtered.append(str(instruction))
            continue
        if (
            arm_params
            and placeholders.difference(arm_params).issubset(available_placeholders)
            and not arm_params.intersection(placeholders)
        ):
            filtered.append(str(instruction))
    return filtered


def _resolve_description_json_path(value: Any) -> Path:
    return OBJECT_DESCRIPTION_DIR / f"{value}.json"


def _replace_placeholders_with_split(
    instruction: str,
    episode_params: Optional[Dict[str, Any]],
    description_split: str,
    rng=None,
) -> str:
    stripped_episode_params = normalize_episode_params(episode_params)
    resolved = str(instruction)

    for key, value in stripped_episode_params.items():
        placeholder = "{" + key + "}"
        value_str = str(value)
        json_path = _resolve_description_json_path(value_str)

        if (("\\" in value_str) or ("/" in value_str)) and not json_path.exists():
            raise FileNotFoundError(
                f"'{json_path}' looks like a description file path, but it does not exist."
            )

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            description_list = json_data.get(description_split, [])
            if not description_list:
                description_list = json_data.get("seen", [])
            if not description_list:
                raise ValueError(f"No descriptions found in '{json_path}'")
            value_str = f"the {_choice(description_list, rng=rng)}"
        elif len(key) == 1 and "a" <= key <= "z":
            value_str = f"the {value_str} arm"

        resolved = resolved.replace(placeholder, value_str)

    return resolved


def replace_placeholders(instruction: str, episode_params: Optional[Dict[str, Any]], rng=None) -> str:
    return _replace_placeholders_with_split(instruction, episode_params, description_split="seen", rng=rng)


def replace_placeholders_unseen(instruction: str, episode_params: Optional[Dict[str, Any]], rng=None) -> str:
    return _replace_placeholders_with_split(instruction, episode_params, description_split="unseen", rng=rng)


def load_task_instructions(task_name: str) -> Dict[str, Any]:
    file_path = TASK_INSTRUCTION_DIR / f"{task_name}.json"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_instruction_bank(value: Any) -> Dict[str, List[str]]:
    if isinstance(value, str):
        text = value.strip()
        return {"seen": ([text] if text else []), "unseen": []}

    if isinstance(value, list):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        return {"seen": normalized, "unseen": []}

    if isinstance(value, dict):
        normalized = {}
        for split in ("seen", "unseen"):
            candidates = value.get(split, [])
            if isinstance(candidates, str):
                candidates = [candidates]
            normalized[split] = [str(item).strip() for item in candidates if str(item).strip()]
        return normalized

    return {"seen": [], "unseen": []}


def _trim_episode_params_for_bank(bank: Dict[str, List[str]], episode_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized_episode_params = normalize_episode_params(episode_params)
    used_placeholders = set()
    for split in ("seen", "unseen"):
        for instruction in bank.get(split, []):
            used_placeholders.update(extract_placeholders(instruction))

    if len(used_placeholders) == 0:
        return {}

    trimmed = {}
    for key in used_placeholders:
        if key not in normalized_episode_params:
            continue
        trimmed["{" + key + "}"] = normalized_episode_params[key]
    return trimmed


def resolve_instruction_list(
    instructions: Iterable[str],
    episode_params: Optional[Dict[str, Any]],
    use_unseen: bool = False,
    max_descriptions: int = 1,
    rng=None,
    shuffle: bool = True,
) -> List[str]:
    filtered = filter_instructions(instructions, episode_params, rng=rng, shuffle=shuffle)
    if len(filtered) == 0 or int(max_descriptions) <= 0:
        return []

    replacer = replace_placeholders_unseen if use_unseen else replace_placeholders
    resolved = []
    while len(resolved) < int(max_descriptions):
        for instruction in filtered:
            if len(resolved) >= int(max_descriptions):
                break
            resolved.append(replacer(instruction, episode_params, rng=rng))
    return resolved


def resolve_instruction_bank(
    bank: Any,
    episode_params: Optional[Dict[str, Any]],
    preferred_splits=("seen", "unseen"),
    max_descriptions: int = 1,
    rng=None,
    shuffle: bool = True,
) -> List[str]:
    normalized_bank = normalize_instruction_bank(bank)
    trimmed_episode_params = _trim_episode_params_for_bank(normalized_bank, episode_params)
    for split in preferred_splits:
        candidates = normalized_bank.get(str(split), [])
        if len(candidates) == 0:
            continue
        resolved = resolve_instruction_list(
            candidates,
            trimmed_episode_params,
            use_unseen=(str(split) == "unseen"),
            max_descriptions=max_descriptions,
            rng=rng,
            shuffle=shuffle,
        )
        if len(resolved) > 0:
            return resolved
    return []
