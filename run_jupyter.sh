#!/usr/bin/env bash
conda init bash
conda activate gospl

# exec the cmd/command in this process, making it pid 1
exec "$@"

SHELL=/bin/bash jupyter notebook "$@"
