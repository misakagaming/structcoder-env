#!/bin/bash
source /opt/python3/venv/base/bin/activate
conda activate
echo "nice"
exec "$@"
