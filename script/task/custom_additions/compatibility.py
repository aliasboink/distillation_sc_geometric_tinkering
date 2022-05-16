import torch
# For the DataSet
from torchdistill.datasets.wrapper import BaseDatasetWrapper
from torchdistill.datasets.wrapper import register_dataset_wrapper
# For the Model
from deep_gcns.examples.sem_seg_dense.architecture import DenseDeepGCN
from deep_gcns.examples.sem_seg_dense.config import OptInit
from torchdistill.models.registry import register_model_func


@register_dataset_wrapper
class GeometricWrapper(BaseDatasetWrapper): 
    def __init__(self, org_dataset):
        super().__init__(org_dataset)

    def __getitem__(self, index):
        data = self.org_dataset.__getitem__(index)
        sample = torch.cat((data.pos.transpose(1, 0).unsqueeze(2), data.x.transpose(1, 0).unsqueeze(2)), 0)
        target = data.y
        supp_dict = dict()
        return sample, target, supp_dict

# To be noted here that the actual model itself also has
# `register_model_class`` called as a decorator in architecture.py
@register_model_func
def dense_deep_gcn(**kwargs):
    opt = OptInit().get_args()
    return DenseDeepGCN(opt)