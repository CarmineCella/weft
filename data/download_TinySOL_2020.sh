#!/usr/bin/env bash
#
# download_TinySOL_2020.sh -- fetch the TinySOL 2020 audio dataset into
# ./TinySOL_2020/
#
# TinySOL is a freely-redistributable subset of OrchideaSOL (Studio On
# Line), a database of isolated instrumental samples covering most of
# the standard orchestra.  Each sample is a single sustained note or
# articulation, recorded under controlled conditions; the metadata
# encodes instrument, dynamic, pitch, and playing technique.
#
# In this project, TinySOL is the source corpus for the audio
# classification examples (sound_classifier, extract_features, the
# audio VAE) and is small enough -- a few hundred MB -- to download
# and use locally.
#
# Source: orch-idea.org, the home of the Orchidea computer-assisted
# orchestration project.  This is the official free distribution
# point for the trimmed dataset.
#
set -euo pipefail

# Move into the directory that holds this script, so the data lands in
# data/TinySOL_2020/ regardless of where the user invoked it from.
cd "$(dirname "$0")"

URL="https://www.orch-idea.org/data/db/TinySOL_2020.zip"
ARCHIVE="TinySOL_2020.zip"
TARGET_DIR="TinySOL_2020"

# Skip if the extracted directory already exists and contains audio.
# The dataset is organised as TinySOL_2020/<Instrument>/<sample>.wav,
# so the presence of any .wav under TARGET_DIR is a sufficient check.
if [ -d "$TARGET_DIR" ] && [ -n "$(find "$TARGET_DIR" -name '*.wav' -print -quit 2>/dev/null)" ]; then
    echo "[skip] $TARGET_DIR/ already present with audio files"
    echo "       $(find "$TARGET_DIR" -name '*.wav' | wc -l) wav files total"
    exit 0
fi

if [ ! -f "$ARCHIVE" ]; then
    echo "[get ] $ARCHIVE"
    curl -fL --progress-bar -o "$ARCHIVE" "$URL"
fi

echo "[unzip] $ARCHIVE"
unzip -q "$ARCHIVE"
rm "$ARCHIVE"

# The zip may extract to TinySOL_2020/ or to some other top-level name;
# normalise to TinySOL_2020/ if needed.  Most likely the archive layout
# matches the directory name, but this guards against a future
# repackaging that nests an extra layer.
if [ ! -d "$TARGET_DIR" ]; then
    # Find the directory that actually got created
    EXTRACTED=$(find . -maxdepth 1 -type d -name 'TinySOL*' -not -path '.' | head -n1)
    if [ -n "$EXTRACTED" ] && [ "$EXTRACTED" != "./$TARGET_DIR" ]; then
        echo "[note] renaming $EXTRACTED -> $TARGET_DIR"
        mv "$EXTRACTED" "$TARGET_DIR"
    fi
fi

echo
echo "TinySOL files are in: $(pwd)/$TARGET_DIR"
N_WAV=$(find "$TARGET_DIR" -name '*.wav' | wc -l | tr -d ' ')
N_INST=$(find "$TARGET_DIR" -maxdepth 2 -type d | wc -l | tr -d ' ')
echo "  $N_WAV wav files across $N_INST directories"
echo
echo "To use with the audio classification examples:"
echo "  cd build"
echo "  ./examples/extract_features ../data/$TARGET_DIR  features.bin"
echo "  ./examples/sound_classifier features.bin"
