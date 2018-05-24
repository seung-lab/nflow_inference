#!/bin/bash
docker build -t seunglab/nflow_inference:worker -f Dockerfile.worker .
docker push seunglab/nflow_inference:worker
