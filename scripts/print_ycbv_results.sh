#!/usr/bin/env bash

cat "$1"/log.txt|tail -n44|head -n18|awk '{print $"'$2'"}'
