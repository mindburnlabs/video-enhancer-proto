#!/usr/bin/env bash
set -euo pipefail

# Safe model fetch utility - clones only permissively licensed models declared in config/model_registry.json
# Requires: git-lfs, jq

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
REGISTRY="$ROOT_DIR/config/model_registry.json"
DEST_DIR="$ROOT_DIR/models/weights"

if ! command -v git &>/dev/null; then
  echo "git is required" >&2; exit 1
fi
if ! command -v git-lfs &>/dev/null && ! command -v git lfs &>/dev/null; then
  echo "git-lfs is required" >&2; exit 1
fi
if ! command -v jq &>/dev/null; then
  echo "jq not found – limited fallback mode will be used"
  JQ_AVAILABLE=0
else
  JQ_AVAILABLE=1
fi

mkdir -p "$DEST_DIR"

if [ "$JQ_AVAILABLE" -eq 1 ]; then
  LICENSE_MODE=$(jq -r '.license_mode // "permissive_only"' "$REGISTRY")
  MODEL_STREAM=$(jq -c '.models[]' "$REGISTRY")
else
  # Fallback defaults
  LICENSE_MODE=${LICENSE_MODE:-permissive_only}
  MODEL_STREAM=''
fi

if [ -n "$MODEL_STREAM" ]; then
  echo "$MODEL_STREAM" | while read -r item; do
    enabled=$(jq -r '.enabled' <<<"$item")
    license=$(jq -r '.license' <<<"$item")
    name=$(jq -r '.name' <<<"$item")
    repo=$(jq -r '.repo' <<<"$item")
    local_path=$(jq -r '.local_path' <<<"$item")

    [ "$enabled" != "true" ] && { echo "Skipping $name (disabled)"; continue; }
    if [ "$LICENSE_MODE" = "permissive_only" ]; then
      case "$license" in
        apache-2.0|Apache-2.0|mit|MIT|bsd-3-clause|BSD-3-Clause) ;;
        *) echo "Skipping $name due to non-permissive or unknown license: $license"; continue;;
      esac
    fi

    target="$ROOT_DIR/$local_path"
    if [ -d "$target/.git" ]; then
      echo "Already fetched: $name at $target"
      continue
    fi

    echo "Fetching $name from $repo → $target"
    mkdir -p "$(dirname "$target")"
    git lfs clone "$repo" "$target"
    echo "OK: $name"
  done
else
  # Fallback: fetch SeedVR2 directly when jq missing and permissive-only
  if [ "$LICENSE_MODE" = "permissive_only" ]; then
    echo "jq unavailable – cloning SeedVR2-3B directly (fallback)"
    mkdir -p "$DEST_DIR"
    if [ ! -d "$ROOT_DIR/models/weights/SeedVR2-3B/.git" ]; then
      git lfs clone https://huggingface.co/ByteDance-Seed/SeedVR2-3B "$ROOT_DIR/models/weights/SeedVR2-3B" || true
    fi
  fi
fi

  [ "$enabled" != "true" ] && { echo "Skipping $name (disabled)"; continue; }
  if [ "$LICENSE_MODE" = "permissive_only" ]; then
    case "$license" in
      apache-2.0|Apache-2.0|mit|MIT|bsd-3-clause|BSD-3-Clause) ;;
      *) echo "Skipping $name due to non-permissive or unknown license: $license"; continue;;
    esac
  fi

  target="$ROOT_DIR/$local_path"
  if [ -d "$target/.git" ]; then
    echo "Already fetched: $name at $target"
    continue
  fi

  echo "Fetching $name from $repo → $target"
  mkdir -p "$(dirname "$target")"
  git lfs clone "$repo" "$target"
  echo "OK: $name"

done

echo "Done. Weights in $DEST_DIR"
