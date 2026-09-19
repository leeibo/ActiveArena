#!/usr/bin/env python3
"""Download the small RMBench assets used by information-gathering tasks.

The task code intentionally references RMBench's prismatic button asset instead
of reimplementing it with a static primitive.  Assets are generated/downloaded
artifacts in this repository, so this script keeps the dependency explicit and
repeatable.
"""

from huggingface_hub import snapshot_download


def main():
    snapshot_download(
        repo_id="TianxingChen/RMBench",
        repo_type="dataset",
        allow_patterns=[
            "objects/005_button/**",
            "objects/006_check_button/**",
            "objects/004_numbercard/**",
        ],
        local_dir="assets",
        resume_download=True,
    )


if __name__ == "__main__":
    main()
