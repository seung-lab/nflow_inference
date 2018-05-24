#!/bin/bash
docker build -t seunglab/nflow_inference:worker_cpu -f Dockerfile.worker.cpu .
docker push seunglab/nflow_inference:worker_cpu
