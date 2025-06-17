#!/usr/bin/env bash
gcloud storage rsync assets/condorgmm_bucket gs://condorgmm-bucket --recursive
