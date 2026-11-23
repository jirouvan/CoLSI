# 第三方来源

| 组件 | 来源 | 本包用途 | 许可文件 |
|---|---|---|---|
| SegNeXt | https://github.com/Visual-Attention-Network/SegNeXt | MSCAN-B与LightHam原作者源码 | third_party/SegNeXt/LICENSE |
| MoGe-3 | https://github.com/microsoft/MoGe | 冻结米制几何及法向 | third_party/MoGe/LICENSE |
| FlexGEMM | https://github.com/JeffreyXiang/FlexGEMM | MoGe稀疏体素细化运行核 | third_party/FlexGEMM/LICENSE |
| utils3d-moge | https://github.com/EasternJournalist/utils3d-moge | MoGe几何工具 | third_party/utils3d-moge/LICENSE |
| PyTorch / torchvision / timm | 通过requirements与CUDA环境安装 | 张量运算、ConvNeXt-Tiny及DropPath | 各已安装发行包的许可 |

MoGe来源commit：`74fbce054ebed49800de42d0ad0e83495065719a`；FlexGEMM：`b2fadb29d41846c7981ade6801ffc689fae119cf`；utils3d-moge：`62f09d58509485564e24d5d9f6aac9ee9ebc0c37`。

原实验源码按最终运行所需符号整理为`colsi`包；CFA、QSF、CCSA、几何描述和融合函数的来源对应记录在`SOURCE_PROVENANCE.json`。大权重置于独立网盘包，发布SHA与原权重SHA记录在`CoLSI-data/weights/manifest.json`。

发布副本对MoGe的可选DINO预训练工具`moge/model/modules/dinov2/utils/utils.py`添加了`torch.load(..., weights_only=True)`；实际完整MoGe加载器使用独立、显式的权重读取路径。
