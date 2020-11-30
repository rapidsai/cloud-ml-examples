#!/usr/bin/env bash
set -e
set -x

gcloud ai-platform jobs submit training $1 --config ./$2
