# Torchdistill with DeepGCN
This is an attempt at a combination of the following repositories:

[torchdistill](https://github.com/yoshitomo-matsubara/torchdistill)

[sc2-benchmark](https://github.com/yoshitomo-matsubara/sc2-benchmark)

[deep_gcns_torch](https://github.com/lightaime/deep_gcns_torch)

The discussion that gives an overview as how this came to be can be found [here](https://github.com/yoshitomo-matsubara/torchdistill/discussions/225).

## Description
### Overview
This is an attempt to integrate `DenseDeepGCN` into the `torchdistill` framework and then further modify it with injected bottlenecks for the purpose of split computing. To this end, the `sc2-benchmark` repository has been cloned and modified, adding `deep_gcns_torch` and utilizing features from the `torchdistill` framework.
### State of the project (Issues, primarily)
You can run the evaluation and the training of a DenseDeepGCN. 
The **IoU calculation** is poorly implemented (on my part) due to how the implementations differ from `dense_gcn` and `torchdstill`. I need to implement it better taking into account the differences.
**Terminal command options** may cause issues as there are two argument parsers. To fix this I will most likely modify the way `OptInit()` works in `DenseDeepGCN` and make `DenseDeepGCN` work with parameters fit for `torchdistill` (it's tidy and makes use of the framework).

## Environment
**NOT WORKING YET!!!!!!!!!!!!!!!!**

To create the environment run:
```
conda create --name torch_distill_gcn --file torchdistill_deep_gcn.txt
```

**READ THIS INSTEAD!!!**

For the creation of the environment I've used the `environment.yaml` file found in `sc2-benchmark` then updating everything `torch` related. 
Then I've installed `torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric` based on my `CUDA` version.

## Testing

After you have downloaded the `S3DIS` dataset and added it to a **datasets** folder in the preferred location, you can run:
```
python test.py --pretrained_model /home/gleip/Desktop/models/sem_seg_dense-res-edge-28-64-ckpt_best_model.pth  --batch_size 1  --data_dir /home/gleip/Desktop/deepgcn/S3DIS
```
