import subprocess
import os
"""
Made this script because I prefer pressing the 'start' button on the IDE
rather than manually typing gradio frontend.py on my terminal each time.
"""


def main():
    app_path = os.path.abspath("frontend.py")

    # Command to run Gradio in reload mode
    cmd = ["gradio", app_path]

    print("Command:", " ".join(cmd))

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
