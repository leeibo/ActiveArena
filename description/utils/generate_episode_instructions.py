import json
from typing import List, Dict, Any
import os
import argparse
import yaml

from instruction_template_utils import (
    extract_placeholders,
    filter_instructions,
    load_task_instructions,
    replace_placeholders,
    replace_placeholders_unseen,
)

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def load_scene_info(task_name: str, setting: str, scene_info_path: str) -> Dict[str, Dict]:
    """Load the scene info from the JSON file in the data directory."""
    file_path = os.path.join(parent_directory, f"../../{scene_info_path}/{task_name}/{setting}/scene_info.json")
    try:
        with open(file_path, "r") as f:
            scene_data = json.load(f)
        return scene_data
    except FileNotFoundError:
        print(f"\033[1mERROR: Scene info file '{file_path}' not found.\033[0m")
        exit(1)
    except json.JSONDecodeError:
        print(f"\033[1mERROR: Scene info file '{file_path}' contains invalid JSON.\033[0m")
        exit(1)


def extract_episodes_from_scene_info(scene_info: Dict) -> List[Dict[str, str]]:
    """Extract episode parameters from scene_info."""
    episodes = []
    for episode_key, episode_data in scene_info.items():
        if "info" in episode_data:
            episodes.append(episode_data["info"])
        else:
            episodes.append(dict())
    return episodes


def save_episode_descriptions(task_name: str, setting: str, generated_descriptions: List[Dict]):
    """Save generated descriptions to output files."""
    output_dir = os.path.join(parent_directory, f"../../data/{task_name}/{setting}/instructions")
    os.makedirs(output_dir, exist_ok=True)

    for episode_desc in generated_descriptions:
        episode_index = episode_desc["episode_index"]
        output_file = os.path.join(output_dir, f"episode{episode_index}.json")

        with open(output_file, "w") as f:
            json.dump(
                {
                    "seen": episode_desc.get("seen", []),
                    "unseen": episode_desc.get("unseen", []),
                },
                f,
                indent=2,
            )

def generate_episode_descriptions(task_name: str, episodes: List[Dict[str, str]], max_descriptions: int = 1000000):
    """
    Generate descriptions for episodes by replacing placeholders in instructions with parameter values.
    For each episode, filter instructions that have matching placeholders and generate up to
    max_descriptions by replacing placeholders with parameter values.
    Now also generates unseen descriptions.
    """
    # Load task instructions
    task_data = load_task_instructions(task_name)
    seen_instructions = task_data.get("seen", [])
    unseen_instructions = task_data.get("unseen", [])

    # Store generated descriptions for each episode
    all_generated_descriptions = []

    # Process each episode
    for i, episode in enumerate(episodes):
        # Filter instructions that have all placeholders matching episode parameters
        filtered_seen_instructions = filter_instructions(seen_instructions, episode)
        filtered_unseen_instructions = filter_instructions(unseen_instructions, episode)

        if filtered_seen_instructions == [] and filtered_unseen_instructions == []:
            print(f"Episode {i}: No valid instructions found")
            continue

        # Generate seen descriptions by replacing placeholders
        seen_episode_descriptions = []
        flag_seen = True
        while (len(seen_episode_descriptions) < max_descriptions and flag_seen and filtered_seen_instructions):
            
            for instruction in filtered_seen_instructions:
                if len(seen_episode_descriptions) >= max_descriptions:
                    flag_seen = False
                    break
                description = replace_placeholders(instruction, episode)
                seen_episode_descriptions.append(description)

        # Generate unseen descriptions by replacing placeholders
        unseen_episode_descriptions = []
        flag_unseen = True
        while (len(unseen_episode_descriptions) < max_descriptions and flag_unseen and filtered_unseen_instructions):
            for instruction in filtered_unseen_instructions:
                if len(unseen_episode_descriptions) >= max_descriptions:
                    flag_unseen = False
                    break
                description = replace_placeholders_unseen(instruction, episode)
                unseen_episode_descriptions.append(description)

        all_generated_descriptions.append({
            "episode_index": i,
            "seen": seen_episode_descriptions,
            "unseen": unseen_episode_descriptions,
        })

    return all_generated_descriptions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate episode descriptions by replacing placeholders")
    parser.add_argument(
        "task_name",
        type=str,
        help="Name of the task (JSON file name without extension)",
    )
    parser.add_argument(
        "setting",
        type=str,
        help="Setting name used to construct the data directory path",
    )
    parser.add_argument(
        "max_num",
        type=int,
        default=100,
        help="Maximum number of descriptions per episode",
    )
    parser.add_argument(
        "config_name",
        type=str,
        nargs="?",
        default="",
        help="Task config yaml name without suffix (optional).",
    )

    args = parser.parse_args()
    config_name = args.config_name if args.config_name else args.setting
    setting_file = os.path.join(
        parent_directory, f"../../task_config/{config_name}.yml"
    )
    with open(setting_file, "r", encoding="utf-8") as f:
        args_dict = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Load scene info and extract episode parameters
    scene_info = load_scene_info(args.task_name, args.setting, args_dict['save_path'])
    episodes = extract_episodes_from_scene_info(scene_info)

    # Generate descriptions
    results = generate_episode_descriptions(args.task_name, episodes, args.max_num)

    # Save results to output files
    save_episode_descriptions(args.task_name, args.setting, results)
    print("Successfully Saved Instructions")
