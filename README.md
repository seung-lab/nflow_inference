# Nflow Inference
Large scale inference for EM images using neural networks


## Parameters
* **-f [--file_name] VALUE** -- params file path. If the file name is given, the consequent command line arguments will be ignored and their values will be expected to be found in the params file in JSON format.

Specifying the parameter file is the prefered method of operating. Example parameter files for CPU, GPU, and GPU distributed computation are given in the repo.

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



## Usage
### Single Machine
Example:
```bash
python3 client.py -f example_params_gpu.json
```
### Distributed
Before runing distributed version, you'll need to set up your AWS secrets. AWS and Google secrets are not included in Dockers or the repo for security reasons. Get in touch with me if you're not sure how to do it.

After your secrets are set up, you can run the following to generate tasks and push them to the queue:
```bash
python3 client.py -f example_params_gpu_distributed.json
```

Then, on as many machines (you have to set up the secrets on those machines) as you want, run:
```bash
python3 worker_distributed.py -f example_params_gpu_distributed.json
```
## Troubleshooting
We use Amazon SQS for task management. You can use the default 'deepalign' queue, or create a custom queue. **Note that you should flush your queues if you want to restart and exeriment.**


