# 基于PP-YOLOV2的电接口状态检测

使用PP-YOLOV2模型对电接口的状态（正常、缺针）进行了检测。

# 一、项目背景

在电子工业工厂中，免不了生产或者使用电接口，而电接口具有针数多，人工检测耗费人力、时间等成本，故想到用机器代替传统人工进行此项工作。

![](https://ai-studio-static-online.cdn.bcebos.com/ab205071d1974c14959fa59cb06a4ab6d6b40d13cb144f4084fc03f565f2e7da)
![](https://ai-studio-static-online.cdn.bcebos.com/5c32d3ae4422470da4b667e7dc8ac0b97f47254bde244227a050e4222329dbfb)


# 二、数据集简介

此项目数据集采用的是VOC格式的数据集，共有120张图片，电接口正常和缺针的状况各占60张，数据处理如下所示：

## 1.数据增强预处理
定义数据处理流程，其中训练和测试需分别定义，训练过程包括了部分测试过程中不需要的数据增强操作，如在本示例中，训练过程使用了MixupImage、RandomDistort、RandomExpand、RandomCrop、RandomHorizontalFlip和BatchRandomResize共6种数据增强方式

```python
import paddlex as pdx
from paddlex import transforms as T

train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=-1), T.RandomDistort(),
    T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]), T.RandomCrop(),
    T.RandomHorizontalFlip(), T.BatchRandomResize(
        target_sizes=[
            320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
            736, 768
        ],
        interp='RANDOM'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        target_size=640, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```


## 2.定义数据集

```python
train_dataset = pdx.datasets.VOCDetection(
    data_dir='DataVOC',
    file_list='DataVOC/train_list.txt',
    label_list='DataVOC/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='DataVOC',
    file_list='DataVOC/val_list.txt',
    label_list='DataVOC/labels.txt',
    transforms=eval_transforms)
```

训练集样本量: 84，验证集样本量: 24





# 三、模型选择和开发

详细说明你使用的算法。此处可细分，如下所示：

## 1.路径聚合网络

![](https://ai-studio-static-online.cdn.bcebos.com/002101b7883a4f61aada1ea938cc86009793abd6f794466982b23794322df45a)


## 2.采用 Mish 激活函数


Mish 激活函数被很多实用的检测器采用，并拥有出色的表现，例如 YOLOv4 和 YOLOv5 都在骨架网络（backbone）的构建中应用 mish 激活函数。而对于 PP-YOLOv2，我们倾向于仍然采用原有的骨架网络，因为它的预训练参数使得网络在 ImageNet 上 top-1 准确率高达 82.4%。所以我们把 mish 激活函数应用在了 detection neck 而不是骨架网络上。

## 3.模型训练
paddlex已经为我们准备好了模型，我们只需配置一些参数即可。

```python
num_classes = len(train_dataset.labels)
model = pdx.det.PPYOLOv2(num_classes=num_classes, backbone='ResNet50_vd_dcn')
model.train(
    num_epochs=120,
    train_dataset=train_dataset,
    train_batch_size=5,
    eval_dataset=eval_dataset,
    pretrain_weights='COCO',
    learning_rate=0.005 / 12,
    warmup_steps=1000,
    warmup_start_lr=0.0,
    lr_decay_epochs=[105, 135, 150],
    save_interval_epochs=1,
    save_dir='output/ppyolov2_r50vd_dcn')
```

    


## 4.模型预测

使用模型进行预测，同时使用pdx.det.visualize将结果可视化

```python
import paddlex as pdx
model = pdx.load_model('output/ppyolov2_r50vd_dcn/best_model')
image_name = 'DataVOC/JPEGImages/79.jpg'
result = model.predict(image_name)
pdx.det.visualize(image_name, result, threshold=0.45, save_dir='./output/ppyolov2_r50vd_dcn')
```

# 四、效果展示

只需将要预测的代码放到模型预测的image_name中运行代码即可，完整代码详见下面链接。
https://aistudio.baidu.com/aistudio/projectdetail/2278112

本项目目前已经可以准确识别电接口正常和缺针的情况，准确度高达70%以上。具体效果见下方图片（OK为正常，Q为缺针）。


![](https://ai-studio-static-online.cdn.bcebos.com/da9eb1e0ac0f4bae836d320ee6c62ac6450fb712265949b9bb4f7e586abc9220)
![](https://ai-studio-static-online.cdn.bcebos.com/a24bcb0983da402d9218ed62e2fedabbacfde42c256f4be2863c6509da652bc5)
![](https://ai-studio-static-online.cdn.bcebos.com/c5aab7d762d846eaa8a38d4745b513cb895b6d8583494227ae6a19d3b922fa28)
![](https://ai-studio-static-online.cdn.bcebos.com/daafb85e4deb4437bfbbf8cb2b800281007e5c2fb6104f2894cbcf6618975eaa)


# 五、总结与升华

## 1.制作数据集要耐心，不要打错标签。
## 2.数据预处理操作不能太少也不能太多，都会影响最后的精度。
## 3.设置模型参数时，num_epochs太少会使得精度不高，太多会导致时间过长或者直接内存溢出训练失败。


**基于PP-YOLOV2的电接口检测，希望可以给工业生产带来方便。**

# 个人简介

我在AI Studio上获得青铜等级，点亮1个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/698604。
