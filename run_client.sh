#!/bin/bash

python3 client.py -m nips5hot8 --net_mip_low 3 --net_mip_high 3 \
      --x_offset 10240 --y_offset 4096 --x_size 57344 --y_size 40960 \
      --source_img 'gs://neuroglancer/pinky40_alignment/prealigned_rechunked' \
      --dest_img 'gs://neuroglancer/nflow_tests/output' --gpu
