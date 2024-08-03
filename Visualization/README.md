## M-images
Visualizing the M-images by:
```
python extract_M-images.py --dataset 'sysu' --gpu 0
```
## t-SNE

1.Save the features in 'tsne.mat' by:
```
python extract_features.py --dataset 'sysu' --gpu0
```
2.Visualizing the feature distribution with t-SNE:
```
python tsne.py --dataset 'sysu' --gpu0
```

## Intra-Inter class distances

1.Save the features in 'tsne.mat' by:
```
python extract_features.py --dataset 'sysu' --gpu0
```
2.Visualizing the intra-inter class distances by:
```
python intra_inter-distance.py
```

## Citation
Most of the code of our Visualization are borrowed from [DEEN](https://github.com/ZYK100/LLCM/blob/main/Visualization/README.md) [1].

## References
[1] Zhang Y, Wang H. Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-identification[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 2153-2162.
