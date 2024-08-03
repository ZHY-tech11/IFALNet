# IFALNet
Pytorch code for paper "Middle modality interactive feature attention learning for Visible-Infrared Person Re-Identification"

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

## References.
[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[3] Zhang Y, Wang H. Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-identification[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 2153-2162.
