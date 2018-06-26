from aligner import Aligner, BoundingBox
from optparse import OptionParser
import json

parser = OptionParser()
parser.add_option("-f", "--params_file", dest="params_file", default=None)

parser.add_option("-m", "--model", dest="model_name", help="model name", metavar="FILE")
parser.add_option("-c", "--crop", dest="crop", type="int", default=128)
parser.add_option("--net_mip_low", dest="net_mip_low", type="int")
parser.add_option("--net_mip_high", dest="net_mip_high", type="int")
parser.add_option("--max_disp", dest="max_disp", type="int", default=2048)
parser.add_option("-p", "--patch_size", dest="patch_size", type="int", default=1024)
parser.add_option("--x_offset", dest="x_offset", type="int", default=0)
parser.add_option("--y_offset", dest="y_offset", type="int", default=0)
parser.add_option("--x_size", dest="x_size", type="int", default=0)
parser.add_option("--y_size", dest="y_size", type="int", default=0)
parser.add_option("-s", "--source_img", dest="source_img", default="")
parser.add_option("-d", "--dest_img", dest="dest_img", default="")

parser.add_option("--stack_start", dest="stack_start", type="int", default=0)
parser.add_option("--stack_end", dest="stack_end", type="int", default=0)
parser.add_option("--move_anchor", dest="move_anchor", default=False, action="store_true")
parser.add_option("--gpu", dest="gpu", default=False, action="store_true")
parser.add_option("-q", "--queue_name", dest="queue_name", default=None)

(options, args) = parser.parse_args()

if not options.params_file is None:
  print ("parsing params file...")
  with open(options.params_file) as f:
    params = json.load(f)

  max_disp   = params["max_disp"]
  model_name = params["model_name"]
  net_crop   = params["crop"]
  mip_range  = (params["net_mip_low"], params["net_mip_high"])
  patch_size = params["patch_size"]
  xy_offset  = (params["x_offset"], params["y_offset"])
  xy_size    = (params["x_size"], params["y_size"])
  source_img = params["source_img"]
  dest_img   = params["dest_img"]

  stack_start = params["stack_start"]
  stack_end   = params["stack_end"]
  move_anchor = params["move_anchor"]
  gpu         = params["gpu"]
  queue_name  = params["queue_name"]
else:
  max_disp    = options.max_disp
  model_name  = options.model_name
  net_crop    = options.crop
  mip_range   = (options.net_mip_low, options.net_mip_high)
  patch_size  = options.patch_size
  xy_offset   = (options.x_offset, options.y_offset)
  xy_size     = (options.x_size, options.y_size)
  source_img  = options.source_img
  dest_img    = options.dest_img

  stack_start = options.stack_start
  stack_end   = options.stack_end
  move_anchor = options.move_anchor
  gpu         = options.gpu
  queue_name  = options.queue_name


model_path = 'model_repository/' + model_name + '.pt'
high_mip_chunk = (patch_size, patch_size)

a = Aligner(model_path, max_disp, net_crop, mip_range, high_mip_chunk,
		        source_img, dest_img, queue_name=queue_name, gpu=gpu)

bbox = BoundingBox(xy_offset[0], xy_offset[0]+xy_size[0],
                   xy_offset[1], xy_offset[1]+xy_size[1], mip=0, max_mip=9)

a.align_ng_stack(stack_start, stack_end, bbox, move_anchor=move_anchor)
