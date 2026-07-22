import argparse
import hashlib
import urllib.request
from pathlib import Path


URL = "https://media.roboflow.com/supervision/video-examples/vehicles.mp4"
MD5 = "8155ff4e4de08cfa25f39de96483f918"


def md5_matches(path: Path) -> bool:
    if not path.exists():
        return False
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == MD5


parser = argparse.ArgumentParser()
parser.add_argument("--output-path", required=True)
args = parser.parse_args()

output_path = Path(args.output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)

if not md5_matches(output_path):
    urllib.request.urlretrieve(URL, output_path)

if not md5_matches(output_path):
    raise RuntimeError(f"Download failed: {output_path}")

print(output_path)
