#!/usr/bin/env bash
set -ex

python core/gdrn_modeling/engine/test_utils.py \
    --result_dir output/gdrn/lmoPbrSO/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e/merged-bop-test/ \
    --result_names resnest50d-online-AugCosyAAEGray-mlBCE-DoubleMask-lmo-pbr-100e-merged-bop-test-iter0_lmo-test.csv \
    --dataset lmo \
    --split test \
    --split-type "" \
    --targets_name test_targets_bop19.json \
    --error_types mspd,mssd,vsd,reS,teS,reteS,ad \
    --render_type cpp
