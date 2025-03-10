# CLIP-Test-time-Counterattacks 🚀
This is the official code of our work:

[CLIP is Strong Enough to Fight Back: Test-time Counterattacks towards Zero-shot Adversarial Robustness of CLIP](https://arxiv.org/abs/2503.03613). Songlong Xing, Zhengyu Zhao, Nicu Sebe. To appear in CVPR 2025.

<p align="center">
  <img src="figures/teaser.png" width="40%" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="figures/fig2b.png" width="40%" />
</p>

## 🛠️ Setup
### Environment
Make sure you have installed conda and use the following commands to get the env ready!
```bash
conda env create -f environment.yml
```
```bash
conda activate TTC
```
```bash
pip install -r requirements.txt
```
### Data preparation
Please download and unzip all the raw datasets into `./data`. It's okay to skip this step because torchvision.datasets will automatically download (most of) them if you don't already have them as you run the code.

## 📬 Updates
 7 Mar 2025: **Please stay tuned for instructions to run the code!**
 
10 Mar 2025: **Updated Setup!**

## 🗂️ Reference
```
@article{xing2025clip,
  title={CLIP is Strong Enough to Fight Back: Test-time Counterattacks towards Zero-shot Adversarial Robustness of CLIP},
  author={Xing, Songlong and Zhao, Zhengyu and Sebe, Nicu},
  journal={arXiv preprint arXiv:2503.03613},
  year={2025}
}
```
