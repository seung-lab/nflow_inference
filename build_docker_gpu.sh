#!/bin/bash
docker build -t seunglab/nflow_inference:gpu -f Dockerfile.gpu .
docker push seunglab/nflow_inference:gpu
