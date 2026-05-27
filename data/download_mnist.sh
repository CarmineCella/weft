#!/usr/bin/env bash
#
# download_mnist.sh -- fetch the four MNIST IDX files into ./mnist/
#
# Uses the AWS S3 mirror that PyTorch's torchvision uses as its primary
# source.  The original yann.lecun.com host is unreliable; this one is
# stable.
#
# Total download is about 11 MB compressed, 52 MB uncompressed.
#
set -euo pipefail

# Move into the directory that holds this script, so the data lands in
# data/mnist/ regardless of where the user invoked the script from.
cd "$(dirname "$0")"
mkdir -p mnist
cd mnist

BASE="https://ossci-datasets.s3.amazonaws.com/mnist"

files=(
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
)

for f in "${files[@]}"; do
    if [ -f "$f" ]; then
        echo "[skip] $f already present"
        continue
    fi
    echo "[get ] $f.gz"
    curl -fL --progress-bar -o "$f.gz" "$BASE/$f.gz"
    echo "[gunzip] $f.gz"
    gunzip "$f.gz"
done

echo
echo "MNIST files are in: $(pwd)"
ls -lh
