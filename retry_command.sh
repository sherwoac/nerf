#!/bin/bash
conda activate nerf
until "$@"; do
  echo -e "$@" failed, retrying in 5 seconds
  sleep 5
done