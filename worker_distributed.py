from aligner import Aligner

model_name = 'jacob'
model_path = 'model_repository/' + model_name + '.pt'
max_displacement = 2048
net_crop  = 128
mip_range = (3, 3)
render_mip = 3
high_mip_chunk = (1024, 1024)
a = Aligner(model_path, max_displacement, net_crop, mip_range, high_mip_chunk, 'gs://neuroglancer/pinky40_alignment/prealigned_rechunked', 'gs://neuroglancer/nflow_tests/' + model_name+'_distributed', queue_name='deepalign')

a.listen_for_tasks()
