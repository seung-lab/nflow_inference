# Nflow Inference
Large scale inference for EM images using neural networks

# Usage
 
## Master
```python3 worker_distributed.py [options]```
Options:

* -f [--file_name] VALUE
* -m [--model_name] VALUE
* -c [--crop] VALUE
* -p [--patch_size] VALUE
* --max_disp VALUE
* -s [--source_img] VALUE
* -d [--dest_img] VALUE
* --x_offset VALUE
* --y_offset VALUE
* --x_size VALUE
* --y_size VALUE
* --stack_start VALUE
* --stack_end VALUE
* --move_anchor -- a bolean flag setting weather the first frame has to be moved from source to destination.
* --gpu . -- a bolean flag setting weather the computation is to be done on a GPU.



## Worker 
```python3 worker_distributed.py [options]```
where the options have to match options used for the Master `client.py`

