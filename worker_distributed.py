from aligner import Aligner
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-m", "--model", dest="model_name",
		  help="model name", metavar="FILE")
parser.add_option("-o", "--output_tag", dest="output_tag", default="")
parser.add_option("-c", "--crop", dest="crop", type="int", default=128)
parser.add_option("--max_disp", dest="max_disp", type="int", default=2048)
parser.add_option("-p", "--patch_size", dest="patch_size", type="int", default=1024)
parser.add_option("-q", "--queue_name", dest="queue_name", default=None)
parser.add_option("--move_anchor", dest="move_anchor", default=False, action="store_true")
parser.add_option("--gpu", dest="gpu", default=False, action="store_true")

(options, args) = parser.parse_args()

model_path = 'model_repository/' + options.model_name + '.pt'
max_displacement = options.max_disp
net_crop  = options.crop
mip_range = (3, 3)
high_mip_chunk = (options.patch_size, options.patch_size)

a = Aligner(model_path, max_displacement, net_crop, mip_range, high_mip_chunk,
		'gs://neuroglancer/pinky40_alignment/prealigned_rechunked',
		'gs://neuroglancer/nflow_tests/' + options.model_name + '_' + options.output_tag,
		queue_name=options.queue_name,
                gpu=options.gpu)

a.listen_for_tasks()
