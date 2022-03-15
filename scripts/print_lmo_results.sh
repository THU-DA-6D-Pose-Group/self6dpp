#!/usr/bin/env bash

cat "$1"/log.txt|tail -n15|head -n13|awk '{print $"'$2'"}'
