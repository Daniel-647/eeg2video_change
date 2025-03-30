# EEG2Video Project Updates

本次更新主要集中在以下几个方面：

## 数据集更新

我们对 SEED-DV 数据集进行了扩展，增加了更多样化的样本，以增强模型的鲁棒性和适用性。


**新增8个大类，每类5个小类**

| 自然景观 |      |      |      |      |      |
| -------- | ---- | ---- | ---- | ---- | ---- |
| 人类行为 |      |      |      |      |      |
| 人脸表情 | 高兴 | 悲伤 | 愤怒 | 惊讶 | 平静 |
| 动物行为 |      |      |      |      |      |
| 人工场景 |      |      |      |      |      |
| 文化符号 |      |      |      |      |      |
| （待定） |      |      |      |      |      |
| 复合事件 |      |      |      |      |      |

## 模型升级

视频生成模块已升级为使用最新的 Wan.2.1 模型，提供了更强的训练和生成能力。

## 代码整理

我们对原有代码进行了整理，使其更易于复现，方便研究人员和开发者进行结果的复现。

# *安装指南*

1. 填写 SEED-DV 的 [许可文件](https://cloud.bcmi.sjtu.edu.cn/sharing/o64PBIsIc) 并 [申请](https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/) 数据集。
2. 下载此仓库：`git clone https://github.com/Daniel-647/eeg2video_change.git`
3. 创建 conda 环境并安装运行代码所需的包。

```bash
conda create -n eegvideo
conda activate eegvideo
pip install -r requirements.txt
```
