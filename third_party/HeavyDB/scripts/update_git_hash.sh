#!/bin/bash

BUILD_DIR=$1
OLD_FILE=$BUILD_DIR/heavyai_git_hash.txt
NEW_FILE=$BUILD_DIR/heavyai_git_hash_new.txt
NEW_HASH=$(git rev-parse --short=10 HEAD 2>/dev/null || echo nogit)
echo "$NEW_HASH" > "$NEW_FILE"
if [ -f "$OLD_FILE" ]; then
  read -r OLD_HASH < "$OLD_FILE"
else
  OLD_HASH=""
fi
if [ "$OLD_HASH" != "$NEW_HASH" ]; then
  echo "$NEW_HASH" > "$OLD_FILE"
fi
