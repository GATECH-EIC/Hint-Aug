# Hint-Aug: Drawing Hints from Foundation Vision Transformers Towards Boosted Few-Shot Parameter-Efficient Tuning</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

**Zhongzhi Yu**, Shang Wu, Yonggan Fu, Shunyao Zhang, and Yingyan (Celine) Lin


Accepted at CVPR 2023. [ [Paper](https://arxiv.org/abs/2304.12520) | [Video](https://www.youtube.com/watch?v=Ben48mkV5JY) ]


## Code Usage

Our code is built on top of [[NOAH]](https://github.com/ZhangYuanhan-AI/NOAH).

### Installation
```
pip install -r requirements.txt
```

### To run our code
All scripts to run our code are stored in the `script` folder, you can run any of them to apply our Hint-Aug on Adapter/LoRA/VPT. For example, here is the command to train an Adapter with our Hint-Aug framework: 
```
sh Adapter.sh
```

## Citation
```
@inproceedings{yu2023hint,
  title={Hint-Aug: Drawing Hints from Foundation Vision Transformers Towards Boosted Few-Shot Parameter-Efficient Tuning},
  author={Yu, Zhongzhi and Wu, Shang and Fu, Yonggan and Zhang, Shunyao and Lin, Yingyan Celine},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11102--11112},
  year={2023}
}
```
