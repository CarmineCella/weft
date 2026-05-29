#!/usr/bin/env bash
#
# download_cifar10.sh -- fetch the CIFAR-10 binary dataset into ./cifar-10-batches-bin/
#
# Uses the official mirror at the University of Toronto.  The binary
# version is the most efficient one to load from C++ -- one byte label
# plus 3072 bytes of pixel data per record, no decompression needed
# beyond the initial tarball.
#
# Total download is about 163 MB compressed, 180 MB uncompressed.
#
set -euo pipefail

# Move into the directory that holds this script, so the data lands in
# data/cifar-10-batches-bin/ regardless of where the user invoked it from.
cd "$(dirname "$0")"

URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
ARCHIVE="cifar-10-binary.tar.gz"
TARGET_DIR="cifar-10-batches-bin"

if [ -d "$TARGET_DIR" ] && [ -f "$TARGET_DIR/data_batch_1.bin" ]; then
    echo "[skip] $TARGET_DIR/ already present with data files"
    ls -lh "$TARGET_DIR"
    exit 0
fi

if [ ! -f "$ARCHIVE" ]; then
    echo "[get ] $ARCHIVE"
    curl -fL --progress-bar -o "$ARCHIVE" "$URL"
fi

echo "[untar] $ARCHIVE"
tar -xzf "$ARCHIVE"
rm "$ARCHIVE"

echo
echo "CIFAR-10 files are in: $(pwd)/$TARGET_DIR"
ls -lh "$TARGET_DIR"
