# image classifier
图像分类模型，有三种训练方式：
- tf2.5.0 + keras 训练
- tf1.15.0 + estimator 训练
- tf1.15.0 + keras网络转成estimator 训练

# 训练数据
- 可以通过执行: python mnist_save_image.py
  数据下载到当前工程目录 resource/mnist/download，解压到resource/mnist/extract/ ，最后保存到resource/mnist/train/ 和 resource/mnist/test/

> 当然可以选择修改config/project.cfg下的配置：
- [MNIST_REGION_CLASSIFIER]
- NET=MOBILENET_V0
- IMAGE_WIDTH=28
- IMAGE_HEIGHT=28
- CHANNELS=1
- SOURCE_IMAGE_TRAIN=resource/mnist/train/
- SOURCE_IMAGE_TEST=resource/mnist/test/
- SOURCE_IMAGE_DOWNLOAD=resource/mnist/download/
- SOURCE_IMAGE_EXTRACT=resource/mnist/extract/

> 用户有自己的训练数据，也可以在config/project.cfg下新增一个project

# 配置
> config/project.cfg
- [MNIST_REGION_CLASSIFIER] -> project
- NET -> 网络结构名
- IMAGE_WIDTH -> input width
- IMAGE_HEIGHT -> input height
- CHANNELS -> input channels
- SOURCE_IMAGE_TRAIN -> 训练数据目录
- SOURCE_IMAGE_TEST -> 测试数据目录
- SOURCE_IMAGE_DOWNLOAD -> 下载数据目录（如果需要）
- SOURCE_IMAGE_EXTRACT -> 解压数据目录（如果需要）

> config/train.cfg
- [TRAIN]
- EPOCHS -> 训练多少轮（keras 训练方式）
- STEPS -> 训练多少步（estimator 训练方式）
- INITIAL_LEARNING_RATE -> 初始学习率
- DECAY_STEPS -> 衰减速度
- DECAY_RATE -> 衰减系数
- MONITOR -> 被监测的数据
- MIN_DELTA -> 在被监测的数据中被认为是提升的最小变化
- PATIENCE -> 在监测质量经过多少轮次没有进度时即停止

# agrs
- project -> 工程名
- net -> 网络结构
- time -> 根据时间生成的训练目录，如20210706-2130（如果之前生成了tfrecord，断掉训练后再次执行命令加上时间则可不用再次生成tfrecord，并且会从执行训练的checkpoint开始）
- steps -> 训练的步数（estimator 训练方式）
- epochs -> 训练多少轮（keras 训练方式）
- gpu -> 使用哪个GPU训练
- debug -> 是否是debug模型
- keras -> 是否使用keras训练

> 比如：
- python main.py -n mobilenet_v0 -k 0 -s 1000000
- python main.py -n mobilenet_v0 -k 1 -e 100000


> 后续只维护tk分支，别的分支不再维护