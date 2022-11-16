import torch
import copy

state_dict = torch.load('/home/dml307/exp/likyoo/project/open-cd/weights/latest-opencd-65645.pth')
delete_keys = ['backbone.ccs.0.mix_norm1.weight', 'backbone.ccs.0.mix_norm1.bias', 'backbone.ccs.0.mix_norm1.running_mean', 
               'backbone.ccs.0.mix_norm1.running_var', 'backbone.ccs.0.mix_norm1.num_batches_tracked', 'backbone.ccs.0.mix_norm2.weight',
               'backbone.ccs.0.mix_norm2.bias', 'backbone.ccs.0.mix_norm2.running_mean', 'backbone.ccs.0.mix_norm2.running_var', 
               'backbone.ccs.0.mix_norm2.num_batches_tracked', 'backbone.ccs.0.mix_layer.weight', 'backbone.ccs.0.mix_layer.bias', 
               'backbone.ccs.1.mix_norm1.weight', 'backbone.ccs.1.mix_norm1.bias', 'backbone.ccs.1.mix_norm1.running_mean', 
               'backbone.ccs.1.mix_norm1.running_var', 'backbone.ccs.1.mix_norm1.num_batches_tracked', 'backbone.ccs.1.mix_norm2.weight', 
               'backbone.ccs.1.mix_norm2.bias', 'backbone.ccs.1.mix_norm2.running_mean', 'backbone.ccs.1.mix_norm2.running_var', 
               'backbone.ccs.1.mix_norm2.num_batches_tracked', 'backbone.ccs.1.mix_layer.weight', 'backbone.ccs.1.mix_layer.bias', 
               'backbone.ccs.2.mix_norm1.weight', 'backbone.ccs.2.mix_norm1.bias', 'backbone.ccs.2.mix_norm1.running_mean', 
               'backbone.ccs.2.mix_norm1.running_var', 'backbone.ccs.2.mix_norm1.num_batches_tracked', 'backbone.ccs.2.mix_norm2.weight', 
               'backbone.ccs.2.mix_norm2.bias', 'backbone.ccs.2.mix_norm2.running_mean', 'backbone.ccs.2.mix_norm2.running_var',
               'backbone.ccs.2.mix_norm2.num_batches_tracked', 'backbone.ccs.2.mix_layer.weight', 'backbone.ccs.2.mix_layer.bias', 
               'backbone.ccs.3.mix_norm1.weight', 'backbone.ccs.3.mix_norm1.bias', 'backbone.ccs.3.mix_norm1.running_mean', 
               'backbone.ccs.3.mix_norm1.running_var', 'backbone.ccs.3.mix_norm1.num_batches_tracked', 'backbone.ccs.3.mix_norm2.weight', 
               'backbone.ccs.3.mix_norm2.bias', 'backbone.ccs.3.mix_norm2.running_mean', 'backbone.ccs.3.mix_norm2.running_var', 
               'backbone.ccs.3.mix_norm2.num_batches_tracked', 'backbone.ccs.3.mix_layer.weight', 'backbone.ccs.3.mix_layer.bias']


old_state = list(state_dict['state_dict'].keys())
for k in old_state:
    if k in delete_keys:
       print(state_dict['state_dict'].pop(k, k))


torch.save(state_dict, '/home/dml307/exp/likyoo/project/open-cd/weights/changer_r18_levir.pth')