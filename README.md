# Nflow Inference
Large scale inference for EM images using neural networks

## Example Walkthrough


## Master
```python3 worker_distributed.py [options]```
Options:

* **-f [--file_name] VALUE** -- params file path. If the file name is given, the consequent command line arguments will be ignored and their values will be expected to be found in the params file in JSON format.
* **-m [--model_name] VALUE** -- the name of the model to use for residual computation. The model has to be put in the `model_repository` folder in this repo.
* **-c [--crop] VALUE** -- the amount of pixels to be cropped from the output of the network
* **-p [--patch_size] VALUE** -- the output patch size of each application of the network. Has to be a multiple of cloudvolume chunk size
* **-s [--source_img] VALUE** -- the ng path to the source volume.
* **--max_disp VALUE** -- maximum displacement expected in the source volume. Higher values make rendering slower.
* **-d [--dest_img] VALUE** -- the ng path to the destination volume. `info` files are created automatically.
* **--x_offset VALUE** -- x offset of the bounding box to be aligned.
* **--y_offset VALUE** -- y offset of the bounding box to be aligned.
* **--x_size VALUE** -- x size of the bounding box to be aligned.
* **--y_size VALUE** -- y offset of the bounding box to be aligned.
* **--stack_start VALUE** -- the index of the first slice in the stack. The first slice will be taken as a reference.
* **--stack_end VALUE** -- the index of the last slice in the stack.
* **--move_anchor** -- a bolean flag setting weather the first frame has to be moved from source to destination.
* **--gpu**  -- a bolean flag setting weather the computation is to be done on a GPU.



## Worker 
```python3 worker_distributed.py [options]```
where the options have to match options used for the Master `client.py`

