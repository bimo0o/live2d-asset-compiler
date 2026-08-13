from pathlib import Path

import torchvision.transforms


def main() -> int:
    site = Path(torchvision.transforms.__file__).parent
    target = site / "functional_tensor.py"
    target.write_text(
        "from torchvision.transforms.functional import *\n",
        encoding="utf-8",
    )
    print(f"patched {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

