# CoLSI-Net

[English](README.md) | 简体中文

CoLSI-Net用于合作式着陆表面的类别与完整可见区域分割。网络以SegNeXt-B的MSCAN-B和LightHam为基础，包含CFA（Cycle-consistent Feature Alignment，循环一致性特征对齐）与QSF（Quality–Semantic Fusion，质量—语义融合），通过CCSA（一致性引导的跨尺度支持注意力）连接。

完整链路为：**RGB、相机内参K、相机系重力g → CoLSI-Net类别与可见mask → 冻结MoGe-3 ViT-L几何描述与ConvNeXt-Tiny RGB-ROI特征 → 824维拼接 → MLP残差融合 → 表面倾角和着陆分级**。

## 数据集下载

**百度网盘：**[下载CoLSI数据集](https://pan.baidu.com/s/1nFXsun_0H7JeOaLMqD3r9Q)

**提取码：**`2026`

下载后，将`CoLSI-data`文件夹放在代码目录旁，采用下面的相邻目录布局。

## 发布目录

```text
CoLSI-code/                 # 本仓库
  colsi/                   # 最终网络、标准链、训练和评价
  configs/                 # 训练协议及相机输入示例
  third_party/             # 运行所需作者源码和原许可
  train.py
  predict.py
  evaluate.py
CoLSI-data/                 # 从网盘下载，放在相邻目录
  CoLSI/                   # CoLSI-Sim、RealMarks、RealCalib与固定划分
  weights/
    colsi_net.pt
    convnext_tiny_roi.pt
    fusion_mlp.pt
    moge3_vitl.pt
    mscan_b_imagenet.pth
    manifest.json
```

训练和评价入口支持`--data-root`；三个入口均支持`--weights-dir`，可将数据和权重放在其他位置。代码目录约数MiB，权重与数据独立下载。

## 环境

使用Python 3.12、CUDA版PyTorch和torchvision。实测环境：Python `3.12.13`、PyTorch `2.12.0.dev20260408+cu128`、torchvision `0.27.0.dev20260407+cu128`、CUDA12.8、RTX5090；关键依赖版本记录在`environment-tested.json`。先按[PyTorch官方安装页](https://pytorch.org/get-started/locally/)选择CUDA版torch/torchvision，再安装：

```bash
python -m pip install -r requirements.txt
```

MoGe、FlexGEMM、utils3d-moge的关键源码随代码发布；推理入口自动加入包内路径。Windows使用triton-windows，Linux使用triton。FlexGEMM采用源码中的Triton后端。神经训练及推理使用CUDA，启动后将GPU名称、容量、模型与输入设备、显存写入输出目录。

## 单图分割及完整链路

```bash
# 仅分割：导出语义图、逐对象类别与完整可见区域
python predict.py --input image.png --segmentation-only --output outputs/segmentation

# 完整链：实际K和g_C写入camera.json
python predict.py --input image.png --camera camera.json --output outputs/landing
```

相机文件格式见`configs/camera.example.json`。K采用原图像素坐标，g_C为相机坐标系重力方向。发布的MoGe FOV适配器要求方形像素和位于`((W−1)/2,(H−1)/2)`的主点；MoGe采用level9、refine3、FP32。示例内参适配640×360，实际输入填写自身标定值。

输出包括`semantic.png`、每个对象的二值可见mask、`prediction.json`及设备记录。按未舍入倾角分级：0°～5°安全着陆，>5°～15°谨慎着陆，>15°不可着陆；角度缺失时倾角和着陆分级留空。

## 训练

```bash
python train.py --output outputs/colsi_net_seed17
python train.py --output outputs/colsi_net_seed17 --resume
```

默认配置见`configs/train.json`：seed17、90轮、物理batch24／有效batch48、BF16、TF32关闭、AdamW、主分支LR0.0006、骨干LR倍率0.1、单段cosine倍率1→0.1。监督为语义CE＋0.1边界BCE，采用原RGB/语义预处理和固定数据划分。发布训练器按整轮保存`last.pt`及优化器、scheduler和RNG，用完整val前景mIoU选择`best.pt`。恢复从最近完整轮开始，结果写入独立输出目录。

网盘同时提供官方MSCAN-B初始化权重供重新训练。标准下游使用现成冻结MoGe、ConvNeXt与MLP权重。

## 评价

```bash
python evaluate.py --task segmentation --split val --output outputs/val_seg
python evaluate.py --task segmentation --split test --output outputs/test_seg
python evaluate.py --task full-chain --split val --output outputs/val_chain
```

评价遵循包内固定canonical顺序、分割B8、NMF临时种子61107、BF16与TF32关闭；图像经OpenCV letterbox到288×512，原图概率恢复沿用原实现。完整链先保存全部预测再读取GT评分；空间匹配IoU0.5，联合F1角误差阈值1°/2°/3°/5°/10°，漏检、错类和无效倾角在MAE90中计90°。RGB与MLP以逐图对象组FP32推理，运行配置随结果记录。单图演示采用B1；正式分割评价采用B8。

已有最终模型结果：完整val前景mIoU98.685922%、test98.719127%；既有标准链test联合F1@5°83.263347%、MAE90 3.385475°。CoLSI-Sim提供主要精确监督；RealMarks为弱标注，RealCalib提供未知棋盘条件诊断与支撑面参考。

## 来源、验证与许可

`SOURCE_PROVENANCE.json`列出最终核心函数的原源及抽取关系。源码抽取保留网络参数名；CPU检查完成888个最终权重张量严格加载、原/发布小夹具242张量相等及前向最大差0，ROI裁剪、48维几何描述、824维特征顺序和MLP亦逐项核对。代码发布整理采用现成权重，科研结果沿用对应原实验。

第三方作者源码与许可见`THIRD_PARTY.md`和`third_party/*/LICENSE`。自研代码的发布许可由作者设置；第三方部分遵循各自原许可。
