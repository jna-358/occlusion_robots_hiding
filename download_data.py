import os
import shutil
import gdown
import zipfile

# Clear data directory
shutil.rmtree("data", ignore_errors=True)

# Pull data
file_id = "1HdXPRbGFoJRAksfdygt1N-3CbeMjJVMZ"
output_file = "data.zip"
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file)
with zipfile.ZipFile(output_file, "r") as zip_ref:
    zip_ref.extractall(".")
os.remove(output_file)
