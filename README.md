# IFALNet
Pytorch code for paper "**Middle modality interactive feature attention learning for Visible-Infrared Person Re-Identification**"

## Results
| Datasets   | Rank@1   | mAP   |
| :------- | :-------: | :-------: | 
| #RegDB[1] | 93.9% | 89.9% | 
| #SYSU-MM01[2] | 77.2% | 73.1% | 
| #LLCM[3]| 67.7% | 50.6% |

The results may have some fluctuation, and fine-tuning hyperparameters may yield better results.

## Datasets
* RegDB [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html).
* SYSU-MM01 [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

  * You need to run `python process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

* LLCM [3]: The LLCM dataset can be downloaded from this [website](https://github.com/ZYK100/LLCM).

## Training
**Train IFALNet by**
```
python train.py --dataset sysu --gpu 0
```
* `--dataset`: which dataset "sysu", "regdb" or "llcm".
* `--gpu`: which gpu to run.

## Testing
**Test a model by**
```
python test.py --dataset 'sysu' --mode 'all' --resume 'model_path' --tvsearch True --gpu 0 
```
* `--dataset`: which dataset "sysu", "regdb" or "llcm".
* `--mode`: "all" or "indoor" (only for sysu dataset).
* `--resume`: the saved model path.
* `--tvsearch`: whether thermal to visible search True or False (only for regdb dataset).
* `--gpu`: which gpu to use.

## Citation
Most of the code of our backbone are borrowed from [AGW](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) [4] and [MMN](https://github.com/ZYK100/MMN) [5].

Thanks a lot for the author's contribution.

Please cite the following paper in your publications if it is helpful:
'''
'''

## References.
[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[3] Zhang Y, Wang H. Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-identification[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 2153-2162.

[4] Ye M, Shen J, Lin G, et al. Deep learning for person re-identification: A survey and outlook[J]. IEEE transactions on pattern analysis and machine intelligence, 2021, 44(6): 2872-2893.

[5] Y. Zhang, Y. Yan, Y. Lu, and H. Wang, “Towards a unified middle modality learning for visible-infrared person re-identification,” in Proceedings of the 29th ACM international conference on multimedia, 2021,pp. 788–796.
