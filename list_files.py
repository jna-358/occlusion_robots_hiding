import glob
import os

if __name__ == "__main__":
    dir = "data"
    files = glob.glob(f"{dir}/**/*", recursive=True)

    # Remove directories
    files = [f for f in files if not os.path.isdir(f)]

    print(f"Found {len(files)} files in {dir} directory:")
    for file in files:
        print(file)
