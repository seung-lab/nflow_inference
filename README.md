# Nflow Inference
Large scale inference for EM images using neural networks

# Usage
 
## Master
```python3 worker_distributed.py [options]```
Options:

* -f [--file_name]
* -m [--model_name]
* -c [--crop]
* -p [--patch_size]
* --max_disp
* -s [--source_img]
* -d [--dest_img]
* --x_offset
* --y_offset
* --x_size
* --y_size



## Worker 
```python3 worker_distributed.py [options]```
where the options have to match options used for the Master `client.py`

