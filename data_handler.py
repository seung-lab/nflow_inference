import numpy as np
from cloudvolume import CloudVolume as cv
from copy import deepcopy
from util import upsample_flow
#from time import time

## Data saving
def save_image_patch(ng_path, float_patch, z, bbox, mip):
  x_range = bbox.x_range(mip=mip)
  y_range = bbox.y_range(mip=mip)
  patch = float_patch[0, :, :, np.newaxis]
  uint_patch = (np.multiply(patch, 256)).astype(np.uint8)
  cv(ng_path, mip=mip, bounded=False, fill_missing=True, autocrop=True,
                                progress=False)[x_range[0]:x_range[1],
                                                y_range[0]:y_range[1], z] = uint_patch

def save_residual_patch(flow, z, bbox, mip, x_paths, y_paths):
  #print ("Saving residual patch {} at mip {}".format(bbox.__str__(mip=0), mip), end='')
  #start = time()
  save_vector_patch(flow, x_paths[mip], y_paths[mip], z, bbox, mip)
  #end = time()
  #print (": {} sec".format(end - start))

def save_vector_patch(flow, x_path, y_path, z, bbox, mip):
  x_res = flow[0, :, :, 0, np.newaxis]
  y_res = flow[0, :, :, 1, np.newaxis]

  x_range = bbox.x_range(mip=mip)
  y_range = bbox.y_range(mip=mip)

  cv(x_path, mip=mip, bounded=False, fill_missing=True, autocrop=True,
                                 progress=False)[x_range[0]:x_range[1],
                                                 y_range[0]:y_range[1], z] = x_res
  cv(y_path, mip=mip, bounded=False, fill_missing=True, autocrop=True,
                                 progress=False)[x_range[0]:x_range[1],
                                                 y_range[0]:y_range[1], z] = y_res


def abs_to_rel_residual(abs_residual, patch, mip):
    x_fraction = patch.x_size(mip=0)
    y_fraction = patch.y_size(mip=0)

    rel_residual = deepcopy(abs_residual)
    rel_residual[0, :, :, 0] /= x_fraction
    rel_residual[0, :, :, 1] /= y_fraction
    return rel_residual

def preprocess_data(data):
    sd = np.squeeze(data)
    ed = np.expand_dims(sd, 0)
    nd = np.divide(ed, float(256.0), dtype=np.float32)
    return nd

def get_image_data(path, z, bbox, mip):
  x_range = bbox.x_range(mip=mip)
  y_range = bbox.y_range(mip=mip)

  data = cv(path, mip=mip, progress=False,
            bounded=False, fill_missing=True)[x_range[0]:x_range[1], y_range[0]:y_range[1], z]
  return preprocess_data(data)

def get_vector_data(path, z, bbox, mip):
  x_range = bbox.x_range(mip=mip)
  y_range = bbox.y_range(mip=mip)

  data = cv(path, mip=mip, progress=False,
            bounded=False, fill_missing=True)[x_range[0]:x_range[1], y_range[0]:y_range[1], z]
  return data

def get_abs_residual(z, bbox, mip, x_path, y_path):
  x = get_vector_data(x_path[mip], z, bbox, mip)[..., 0, 0]
  y = get_vector_data(y_path[mip], z, bbox, mip)[..., 0, 0]
  result = np.stack((x, y), axis=2)
  return np.expand_dims(result, axis=0)

def get_rel_residual(z, bbox, mip, x_path, y_path):
  x = get_vector_data(x_path, z, bbox, mip)[..., 0, 0]
  y = get_vector_data(y_path, z, bbox, mip)[..., 0, 0]
  abs_res = np.stack((x, y), axis=2)
  abs_res = np.expand_dims(abs_res, axis=0)
  rel_res = abs_to_rel_residual(abs_res, bbox, mip)
  return rel_res


def get_aggregate_rel_flow( z, bbox, res_mip_range, mip, low_mip, high_mip, x_paths, y_paths):
  result = np.zeros((1, bbox.x_size(mip), bbox.y_size(mip), 2), dtype=np.float32)
  start_mip = max(res_mip_range[0], low_mip)
  end_mip   = min(res_mip_range[1], high_mip)

  for res_mip in range(start_mip, end_mip + 1):
    scale_factor = 2**(res_mip - mip)

    rel_res = get_rel_residual(z, bbox, res_mip, x_paths[res_mip], y_paths[res_mip])
    up_rel_res = upsample_flow(rel_res, scale_factor)

    result += up_rel_res

  return result

