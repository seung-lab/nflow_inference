from cloudvolume import CloudVolume as cv
from pathos.multiprocessing import ProcessPool, ThreadPool
from time import time, sleep
from copy import deepcopy
import numpy as np
import os
import json

from process import Process
from util import crop, warp, upsample_flow, downsample_mip
from boundingbox import BoundingBox, deserialize_bbox
from task_handler import TaskHandler, make_residual_task_message, make_render_task_message, make_copy_task_message, make_downsample_task_message
import data_handler

class Aligner:
  def __init__(self, model_path, max_displacement, crop,
               mip_range, high_mip_chunk, src_ng_path, dst_ng_path,
               render_low_mip=2, render_high_mip=8, is_Xmas=False, threads = 10,
               max_chunk = (1024, 1024), max_render_chunk = (2048*2, 2048*2),
               queue_name=None, gpu=False):

    if queue_name != None:
      self.task_handler = TaskHandler(queue_name)
      self.distributed  = True
    else:
      self.task_handler = None
      self.distributed  = False
    self.threads = 10
    self.process_high_mip = mip_range[1]
    self.process_low_mip  = mip_range[0]
    self.render_low_mip  = render_low_mip
    self.render_high_mip = render_high_mip
    self.high_mip = max(self.render_high_mip, self.process_high_mip)
    self.high_mip_chunk  = high_mip_chunk
    self.max_chunk       = max_chunk
    self.max_render_chunk = max_render_chunk

    self.max_displacement = max_displacement
    self.crop_amount = crop
    self.org_ng_path = src_ng_path
    self.src_ng_path = self.org_ng_path

    self.dst_ng_path = os.path.join(dst_ng_path, 'image')
    self.tmp_ng_path = os.path.join(dst_ng_path, 'intermediate')


    self.res_ng_paths  = [os.path.join(dst_ng_path, 'vec/{}'.format(i))
                                                    for i in range(self.process_high_mip + 10)] #TODO
    self.x_res_ng_paths = [os.path.join(r, 'x') for r in self.res_ng_paths]
    self.y_res_ng_paths = [os.path.join(r, 'y') for r in self.res_ng_paths]

    self.net = Process(model_path, is_Xmas=is_Xmas, cuda=gpu)

    self.dst_chunk_sizes   = []
    self.dst_voxel_offsets = []
    self.vec_chunk_sizes   = []
    self.vec_voxel_offsets = []
    self.vec_total_sizes   = []
    self._create_info_files(max_displacement)
    self.pool = ThreadPool(threads)

    #if not chunk_size[0] :
    #  raise Exception("The chunk size has to be aligned with ng chunk size")

  def set_chunk_size(self, chunk_size):
    self.high_mip_chunk = chunk_size

  def _create_info_files(self, max_offset):
    tmp_dir = "/tmp/{}".format(os.getpid())
    nocache_f = '"Cache-Control: no-cache"'

    os.system("mkdir {}".format(tmp_dir))

    src_info = cv(self.src_ng_path).info
    dst_info = deepcopy(src_info)

    ##########################################################
    #### Create dest info file
    ##########################################################
    chunk_size = dst_info["scales"][0]["chunk_sizes"][0][0]
    dst_size_increase = max_offset
    if dst_size_increase % chunk_size != 0:
      dst_size_increase = dst_size_increase - (dst_size_increase % max_offset) + chunk_size
    scales = dst_info["scales"]
    for i in range(len(scales)):
      scales[i]["voxel_offset"][0] -= int(dst_size_increase / (2**i))
      scales[i]["voxel_offset"][1] -= int(dst_size_increase / (2**i))

      scales[i]["size"][0] += int(dst_size_increase / (2**i))
      scales[i]["size"][1] += int(dst_size_increase / (2**i))

      x_remainder = scales[i]["size"][0] % scales[i]["chunk_sizes"][0][0]
      y_remainder = scales[i]["size"][1] % scales[i]["chunk_sizes"][0][1]

      x_delta = 0
      y_delta = 0
      if x_remainder != 0:
        x_delta = scales[i]["chunk_sizes"][0][0] - x_remainder
      if y_remainder != 0:
        y_delta = scales[i]["chunk_sizes"][0][1] - y_remainder

      scales[i]["size"][0] += x_delta
      scales[i]["size"][1] += y_delta

      scales[i]["size"][0] += int(dst_size_increase / (2**i))
      scales[i]["size"][1] += int(dst_size_increase / (2**i))

      #make it slice-by-slice writable
      scales[i]["chunk_sizes"][0][2] = 1

      self.dst_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.dst_voxel_offsets.append(scales[i]["voxel_offset"])

    cv(self.dst_ng_path, info=dst_info).commit_info()
    cv(self.tmp_ng_path, info=dst_info).commit_info()

    ##########################################################
    #### Create vec info file
    ##########################################################
    vec_info = deepcopy(src_info)
    vec_info["data_type"] = "float32"
    scales = deepcopy(vec_info["scales"])
    for i in range(len(scales)):
      self.vec_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.vec_voxel_offsets.append(scales[i]["voxel_offset"])
      self.vec_total_sizes.append(scales[i]["size"])

      cv(self.x_res_ng_paths[i], info=vec_info).commit_info()
      cv(self.y_res_ng_paths[i], info=vec_info).commit_info()

  def check_all_params(self):
    return True

  def get_upchunked_bbox(self, bbox, ng_chunk_size, offset, mip):
    raw_x_range = bbox.x_range(mip=mip)
    raw_y_range = bbox.y_range(mip=mip)

    x_chunk = ng_chunk_size[0]
    y_chunk = ng_chunk_size[1]

    x_offset = offset[0]
    y_offset = offset[1]

    x_remainder = ((raw_x_range[0] - x_offset) % x_chunk)
    y_remainder = ((raw_y_range[0] - y_offset) % y_chunk)

    x_delta = 0
    y_delta = 0
    if x_remainder != 0:
      x_delta =  x_chunk - x_remainder
    if y_remainder != 0:
      y_delta =  y_chunk - y_remainder

    calign_x_range = [raw_x_range[0] + x_delta, raw_x_range[1]]
    calign_y_range = [raw_y_range[0] + y_delta, raw_y_range[1]]

    x_start = calign_x_range[0] - x_chunk
    y_start = calign_y_range[0] - y_chunk

    x_start_m0 = x_start * 2**mip
    y_start_m0 = y_start * 2**mip

    result = BoundingBox(x_start_m0, x_start_m0 + bbox.x_size(mip=0),
                         y_start_m0, y_start_m0 + bbox.y_size(mip=0),
                         mip=0, max_mip=self.process_high_mip)
    return result

  def break_into_chunks(self, bbox, ng_chunk_size, offset, mip, render=False):
    chunks = []
    raw_x_range = bbox.x_range(mip=mip)
    raw_y_range = bbox.y_range(mip=mip)

    x_chunk = ng_chunk_size[0]
    y_chunk = ng_chunk_size[1]

    x_offset = offset[0]
    y_offset = offset[1]

    x_remainder = ((raw_x_range[0] - x_offset) % x_chunk)
    y_remainder = ((raw_y_range[0] - y_offset) % y_chunk)

    x_delta = 0
    y_delta = 0
    if x_remainder != 0:
      x_delta =  x_chunk - x_remainder
    if y_remainder != 0:
      y_delta =  y_chunk - y_remainder

    calign_x_range = [raw_x_range[0] - x_remainder, raw_x_range[1]]
    calign_y_range = [raw_y_range[0] - y_remainder, raw_y_range[1]]

    x_start = calign_x_range[0] - x_chunk
    y_start = calign_y_range[0] - y_chunk

    if (self.process_high_mip > mip):
        high_mip_scale = 2**(self.process_high_mip - mip)
    else:
        high_mip_scale = 1

    processing_chunk = (int(self.high_mip_chunk[0] * high_mip_scale),
                        int(self.high_mip_chunk[1] * high_mip_scale))
    if not render and (processing_chunk[0] > self.max_chunk[0]
                      or processing_chunk[1] > self.max_chunk[1]):
      processing_chunk = self.max_chunk
    elif render and (processing_chunk[0] > self.max_render_chunk[0]
                     or processing_chunk[1] > self.max_render_chunk[1]):
      processing_chunk = self.max_render_chunk

    for xs in range(calign_x_range[0], calign_x_range[1], processing_chunk[0]):
      for ys in range(calign_y_range[0], calign_y_range[1], processing_chunk[1]):
        chunks.append(BoundingBox(xs, xs + processing_chunk[0],
                                 ys, ys + processing_chunk[0],
                                 mip=mip, max_mip=self.high_mip))

    return chunks


  ## Residual computation
  def run_net_test(self, s, t, mip):
    abs_residual = self.net.process(s, t, mip)

  def compute_residual_patch(self, source_z, target_z, out_patch_bbox, mip):
    #print ("Computing residual for {}".format(out_patch_bbox.__str__(mip=0)),
    #        end='', flush=True)
    precrop_patch_bbox = deepcopy(out_patch_bbox)
    precrop_patch_bbox.uncrop(self.crop_amount, mip=mip)

    if mip == self.process_high_mip:
      src_patch = data_handler.get_image_data(self.src_ng_path, source_z, precrop_patch_bbox, mip)
    else:
      src_patch = data_handler.get_image_data(self.tmp_ng_path, source_z, precrop_patch_bbox, mip)

    tgt_patch = data_handler.get_image_data(self.dst_ng_path, target_z, precrop_patch_bbox, mip)

    abs_residual = self.net.process(src_patch, tgt_patch, mip, crop=self.crop_amount)
    #rel_residual = precrop_patch_bbox.spoof_x_y_residual(1024, 0, mip=mip,
    #                        crop_amount=self.crop_amount)
    data_handler.save_residual_patch(abs_residual, source_z, out_patch_bbox, mip,
                                     self.x_res_ng_paths, self.y_res_ng_paths)


  ## Patch manipulation
  def warp_patch(self, ng_path, z, bbox, res_mip_range, mip):
    influence_bbox =  deepcopy(bbox)
    influence_bbox.uncrop(self.max_displacement, mip=0)

    agg_flow = influence_bbox.identity(mip=mip)
    agg_flow = np.expand_dims(agg_flow, axis=0)
    agg_res  = data_handler.get_aggregate_rel_flow(z, influence_bbox, res_mip_range, mip,
                                         self.process_low_mip, self.process_high_mip,
                                         self.x_res_ng_paths, self.y_res_ng_paths)
    agg_flow += agg_res

    raw_data = data_handler.get_image_data(ng_path, z, influence_bbox, mip)
    #no need to warp if flow is identity
    #warp introduces noise
    if not influence_bbox.is_identity_flow(agg_flow, mip=mip):
      warped   = warp(raw_data, agg_flow)
    else:
      #print ("not warping")
      warped = raw_data[0]

    mip_disp = int(self.max_displacement / 2**mip)
    cropped  = crop(warped, mip_disp)
    result   = data_handler.preprocess_data(cropped * 256)
    #preprocess divides by 256 and puts it into right dimensions
    #this data range is good already, so mult by 256
    data_handler.save_image_patch(self.dst_ng_path, result, z, bbox, mip)

  def downsample_patch(self, ng_path, z, bbox, mip):
    in_data = data_handler.get_image_data(ng_path, z, bbox, mip - 1)
    result  = downsample_mip(in_data)
    return result

  ## High level services
  def copy_section(self, source, dest, z, bbox, mip):
    print ("moving section {} mip {} to dest".format(z, mip), end='', flush=True)
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst_chunk_sizes[mip],
                                    self.dst_voxel_offsets[mip], mip=mip, render=True)
    #for patch_bbox in chunks:
    if self.distributed and len(chunks) > self.threads * 2:
      for i in range(0, len(chunks), self.threads):
        task_patches = []
        for j in range(i, min(len(chunks), i + self.threads)):
          task_patches.append(chunks[j])

        copy_task = make_copy_task_message(z, source, dest, task_patches, mip=mip)
        self.task_handler.send_message(copy_task)

      while not self.task_handler.is_empty():
        sleep(1)
    else:
      def chunkwise(patch_bbox):
        raw_patch = data_handler.get_image_data(source, z, patch_bbox, mip)
        data_handler.save_image_patch(dest, raw_patch, z, patch_bbox, mip)

      self.pool.map(chunkwise, chunks)

    end = time()
    print (": {} sec".format(end - start))

  def prepare_source(self, z, bbox, mip):
    print ("Prerendering mip {}".format(mip),
           end='', flush=True)
    start = time()

    chunks = self.break_into_chunks(bbox, self.dst_chunk_sizes[mip],
                                    self.dst_voxel_offsets[mip], mip=mip, render=True)
    #for patch_bbox in chunks:
    def chunkwise(patch_bbox):
      self.warp_patch(self.src_ng_path, z, patch_bbox, (mip + 1, self.process_high_mip), mip)

    self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def render(self, z, bbox, mip):
    print ("Rendering mip {}".format(mip),
              end='', flush=True)
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst_chunk_sizes[mip],
                                    self.dst_voxel_offsets[mip], mip=mip, render=True)

    if self.distributed:
      for i in range(0, len(chunks), self.threads):
        task_patches = []
        for j in range(i, min(len(chunks), i + self.threads)):
          task_patches.append(chunks[j])

        render_task = make_render_task_message(z, task_patches, mip=mip)
        self.task_handler.send_message(render_task)

      while not self.task_handler.is_empty():
        sleep(1)
    else:
      def chunkwise(patch_bbox):
        print ("Rendering {} at mip {}".format(patch_bbox.__str__(mip=0), mip),
                end='', flush=True)
        self.warp_patch(self.src_ng_path, z, patch_bbox, (mip, self.process_high_mip), mip)
      self.pool.map(chunkwise, chunks)

    end = time()
    print (": {} sec".format(end - start))

  def render_section_all_mips(self, z, bbox):
    #total_bbox = self.get_upchunked_bbox(bbox, self.dst_chunk_sizes[self.process_high_mip],
    #                                           self.dst_voxel_offsets[self.process_high_mip],
    #                                           mip=self.process_high_mip)
    self.render(z, bbox, self.render_low_mip)
    self.downsample(z, bbox, self.render_low_mip, self.render_high_mip)

  def downsample(self, z, bbox, source_mip, target_mip):
    for m in range(source_mip+1, target_mip + 1):
      print ("Downsampleing mip {}: ".format(m),
              end='', flush=True)
      start = time()
      chunks = self.break_into_chunks(bbox, self.dst_chunk_sizes[m],
                                      self.dst_voxel_offsets[m], mip=m, render=True)

      #for patch_bbox in chunks:
      if self.distributed and len(chunks) > self.threads * 2:
        for i in range(0, len(chunks), self.threads):
          task_patches = []
          for j in range(i, min(len(chunks), i + self.threads)):
            task_patches.append(chunks[j])

          downsample_task = make_downsample_task_message(z, task_patches, mip=m)
          self.task_handler.send_message(downsample_task)

        while not self.task_handler.is_empty():
          sleep(1)
      else:
        def chunkwise(patch_bbox):
          print ("Downsampling {} to mip {}".format(patch_bbox.__str__(mip=0), m))
          downsampled_patch = self.downsample_patch(self.dst_ng_path, z, patch_bbox, m)
          data_handler.save_image_patch(self.dst_ng_path, downsampled_patch, z, patch_bbox, m)
        self.pool.map(chunkwise, chunks)
      end = time()
      print (": {} sec".format(end - start))

  def compute_section_pair_residuals(self, source_z, target_z, bbox):
    for m in range(self.process_high_mip,  self.process_low_mip - 1, -1):
      print ("Running net at mip {}".format(m),
                                end='', flush=True)
      start = time()
      chunks = self.break_into_chunks(bbox, self.vec_chunk_sizes[m],
                                      self.vec_voxel_offsets[m], mip=m)
      for patch_bbox in chunks:
      #FIXME Torch runs out of memory
      #FIXME batchify download and upload
        if self.distributed:
          residual_task = make_residual_task_message(source_z, target_z, patch_bbox, mip=m)
          self.task_handler.send_message(residual_task)
        else:
          self.compute_residual_patch(source_z, target_z, patch_bbox, mip=m)
      #self.pool.map(chunkwise, chunks)
      if self.distributed:
        while not self.task_handler.is_empty():
          sleep(1)
      end = time()
      print (": {} sec".format(end - start))
      if m > self.process_low_mip:
        self.prepare_source(source_z, bbox, m - 1)



  ## Whole stack operations
  def align_ng_stack(self, start_section, end_section, bbox, move_anchor=True):
    if not self.check_all_params():
      raise Exception("Not all parameters are set")
    #if not bbox.is_chunk_aligned(self.dst_ng_path):
    #  raise Exception("Have to align a chunkaligned size")
    start = time()
    if move_anchor:
      for m in range(self.render_low_mip, self.high_mip):
        self.copy_section(self.src_ng_path, self.dst_ng_path, start_section, bbox, mip=m)

    for z in range(start_section, end_section):
      self.compute_section_pair_residuals(z + 1, z, bbox)
      self.render_section_all_mips(z + 1, bbox)
    end = time()
    print ("Total time for aligning {} slices: {}".format(end_section - start_section,
                                                          end - start))
  ## Distribution
  def handle_residual_task(self, message):
    source_z = message['source_z']
    target_z = message['target_z']
    patch_bbox = deserialize_bbox(message['patch_bbox'])
    mip = message['mip']
    self.compute_residual_patch(source_z, target_z, patch_bbox, mip)

  def handle_render_task(self, message):
    z = message['z']
    patches  = [deserialize_bbox(p) for p in message['patches']]
    mip = message['mip']
    def chunkwise(patch_bbox):
      print ("Rendering {} at mip {}".format(patch_bbox.__str__(mip=0), mip),
              end='', flush=True)
      self.warp_patch(self.src_ng_path, z, patch_bbox, (mip, self.process_high_mip), mip)
    self.pool.map(chunkwise, patches)

  def handle_copy_task(self, message):
    z = message['z']
    patches  = [deserialize_bbox(p) for p in message['patches']]
    mip = message['mip']
    source = message['source']
    dest = message['dest']
    def chunkwise(patch_bbox):
      raw_patch = data_handler.get_image_data(source, z, patch_bbox, mip)
      data_handler.save_image_patch(dest, raw_patch, z, patch_bbox, mip)
    self.pool.map(chunkwise, patches)

  def handle_downsample_task(self, message):
    z = message['z']
    patches  = [deserialize_bbox(p) for p in message['patches']]
    mip = message['mip']
    def chunkwise(patch_bbox):
      downsampled_patch = self.downsample_patch(self.dst_ng_path, z, patch_bbox, mip)
      data_handler.save_image_patch(self.dst_ng_path, downsampled_patch, z, patch_bbox, mip)
    self.pool.map(chunkwise, patches)

  def handle_task_message(self, message):
    #message types:
    # -compute residual
    # -prerender future target
    # -render final result
    # -downsample
    # -copy

    #import pdb; pdb.set_trace()
    body = json.loads(message['Body'])
    task_type = body['type']
    if task_type == 'residual_task':
      self.handle_residual_task(body)
    elif task_type == 'render_task':
      self.handle_render_task(body)
    elif task_type == 'copy_task':
      self.handle_copy_task(body)
    elif task_type == 'downsample_task':
      self.handle_downsample_task(body)
    else:
      raise Exception("Unsupported task type '{}' received from queue '{}'".format(task_type,
                                                                 self.task_handler.queue_name))

  def listen_for_tasks(self):
    while (True):
      message = self.task_handler.get_message()
      if message != None:
        print ("Got a job")
        s = time()
        self.handle_task_message(message)
        self.task_handler.delete_message(message)
        e = time()
        print ("Done: {} sec".format(e - s))
      else:
        sleep(3)
        print ("Waiting for jobs...")
