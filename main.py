from pathlib import Path

from slam_system.run_realtime import run_realtime

def main():
    run_realtime(image_dir=Path("./pre_processing/frames"), fps=25, bb_interval=5, processing_interval=5)

if __name__ == "__main__":
    main()
