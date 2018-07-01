#!/bin/bash
nvidia-docker run -v ~/.cloudvolume/secrets:/root/.cloudvolume/secrets -it seunglab/nflow_inference:gpu bash
