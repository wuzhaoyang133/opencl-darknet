# Tiny_dark
## 运行环境

- [ ] windows + VS2015 + AMD APP SDK 2.9 (opencl)
- [x] ubuntu + cuda + opencl
- [ ] centos + opencl

## 硬件支持

- [ ] AMD显卡
- [x] NVIDIA GPU
- [ ] FPGA

## 主要工程目录

```angular2html
Tiny_dark(ubuntu+nvidia)
    |--data
    |--demo
    |--obj
    |--src
        |--layers
        |     |--common
        |--net
        |--utils
        |--darknet.c
    |--core
    |--darknet
    |--Makefile
```
## demo流程
##### demo目录结构
```
demo
 |--backup
 |	  |--lenet_cifar10.weight——权重文件
 |--cfg
 |   |--cifar10.data——数据的相关信息
 |	 |--labels.txt——各个类别的名称
 |	 |--lenet.cfg——网络结构
 |--（.png）文件——测试图片
```
##### 工程执行命令
使用lenet完成CIFAR10分类。
```
$ make
$ ./darknet classifier predict demo/cfg/cifar10.data demo/cfg/lenet.cfg demo/backup/lenet_cifar10.weights
```
执行过程中，选择图片路径为：
```
enter image path： demo/ship.png
```


## src目录分析
##### darknet.c
<p> 该脚本为主文件，包含main函数。

##### common
```angular2html
common
    |--activation.c  激活函数
    |--blas.c        基本线性代数库
    |--col2im.c      col转图片
    |--gemm.c        矩阵乘法
    |--im2col.c      图片转col
    |--oclutils.c    opencl支持相关
```

##### util
<p>宿主机功能支持依赖项

##### net
```angular2html
net
 |--classifier      分类器
 |--layer           层对象定义及功能
 |--network         网络对象定义及功能
 |--parser          配置文件解析
```

##### layers
<p> 卷积层、池化层等的具体实现


