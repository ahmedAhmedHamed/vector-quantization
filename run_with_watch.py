import subprocess
import os
from typing import List
"""
Made this script because I prefer pressing the 'start' button on the IDE
rather than manually typing gradio frontend.py on my terminal each time.
"""


def main() -> None:
    app_path: str = os.path.abspath("frontend/frontend.py")

    cmd: List[str] = ["py", "-m", "gradio", app_path]

    print("Command:", " ".join(cmd))

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
