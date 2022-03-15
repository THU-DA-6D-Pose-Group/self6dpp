#!/usr/bin/env bash

cat "$1"/log.txt|tail -n18|awk '{print $2}'
