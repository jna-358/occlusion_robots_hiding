import zipfile
import glob
import shutil

archive_name = "data.zip"

if __name__ == "__main__":
    # Cleanup
    shutil.rmtree(archive_name, ignore_errors=True)

    # Get files to archive
    file_overview = "data.txt"
    with open(file_overview, "r") as f:
        files = f.readlines()
    files = [f.strip() for f in files]

    # Create zip file
    with zipfile.ZipFile("data.zip", "w") as zipf:
        for file in files:
            zipf.write(file)
    print(f"Created zipfile")
