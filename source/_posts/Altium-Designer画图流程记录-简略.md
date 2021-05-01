---
title: Altium Designer画图流程记录(简略)
date: 2021-04-30 22:47:47
tags: PCB
categories: 硬件
---

之前看过博主[半夜中暑的羊](https://space.bilibili.com/384891)的原理图以及PCB绘制[教程](https://www.bilibili.com/video/BV1JE41127Ks)，收获颇丰，但是视频比较长而且分为多集，每次有些地方忘记时要去找到很不方便，因此趁着这次画一个简单转接版的机会，把视频中的内容整理成文字版，方便之后查找和复习。这里面只涉及基本的两层版的基本绘制。

共有以下几部分内容：

1. 原理图库
2. 核心模块的原理图
3. 为原理图添加封装
4. PCB布局
5. PCB布线
6. PCB敷铜
7. PCB生成BOM及投板

<!-- more -->

## PCB各层简介

参考自：[PCB各层介绍](https://blog.csdn.net/qq_29373285/article/details/85339076)

### Mechanical layer(机械层)

Altium Designer提供了16个机械层，它一般用于设置电路板的外形尺寸，数据标记，对齐标记，装配说明以及其它的机械信息。这些信息因设计公司或PCB制造厂家的要求而有所不同。另外，机械层可以附加在其它层上一起输出显示。

![MechanicalLayer](https://raw.githubusercontent.com/eternityqjl/blogGallery/master/blog/MechanicalLayer.PNG)

### Keep out layer(禁止布线层)

用于定义在电路板上能够有效放置元件和布线的区域。在该层绘制一个封闭区域作为布线有效区，在该区域外是不能自动布局和布线的。

![KeepOutLayer](https://raw.githubusercontent.com/eternityqjl/blogGallery/master/blog/KeepOutLayer.PNG)

### Signal layer(信号层)

信号层主要用于布置电路板上的导线。Altium Designer提供了32个信号层，包括Top layer(顶层)，Bottom layer(底层)和32个内电层。

![SignalLayer](https://raw.githubusercontent.com/eternityqjl/blogGallery/master/blog/SignalLayer.PNG)

### Internal plane layer(内部电源/接地层)

Altium Designer提供了32个内部电源层/接地层。该类型的层仅用于多层板，主要用于布置电源层和接地层。我们称双层板，四层板，六层板，一般指信号层和内部电源/接地层的数目。

### Silkscreen layer(丝印层) 

丝印层主要用于放置印制信息，如元件的轮廓和标注，各种注释字符等。Altium Designer提供了Top Overlay（顶层丝印层）和Bottom Overlay（底层丝印层）两个丝印层。

![SilkscreenLayer](https://raw.githubusercontent.com/eternityqjl/blogGallery/master/blog/SilkscreenLayer.PNG)

### Solder mask layer(阻焊层) 

在焊盘以外的各部位涂覆一层涂料，我们通常用的有绿油、蓝油等，用于阻止这些部位上锡。阻焊层用于在设计过程中**匹配焊盘**，是**自动产生**的。阻焊层是负片输出，阻焊层的地方不盖油，其他地方盖油。Altium Designer提供了**Top Solder(顶层)**和**Bottom Solder(底层)**两个阻焊层。

![SolderMaskLayer](https://raw.githubusercontent.com/eternityqjl/blogGallery/master/blog/SolderMaskLayer.PNG)

### Paste mask layer(助焊层，SMD贴片层) 

它和阻焊层的作用相似，不同的是在机器焊接时对应的表面粘贴式元件的焊盘。Altium Designer提供了Top Paste(顶层助焊层)和Bottom Paste(底层助焊层)两个助焊层。主要针对PCB板上的SMD元件。在将SMD元件贴PCB板上以前，必须在每一个SMD焊盘上先涂上锡膏，在涂锡用的钢网就一定需要这个Paste Mask文件,菲林胶片才可以加工出来。Paste Mask层的Gerber输出最重要的一点要清楚，即这个层主要针对SMD元件，同时将这个层与上面介绍的Solder Mask作一比较，弄清两者的不同作用，因为从菲林胶片图中看这两个胶片图很相似。

![PasteMaskLayer](https://raw.githubusercontent.com/eternityqjl/blogGallery/master/blog/PasteMaskLayer.PNG)

### Multi layer(多层) 

电路板上焊盘和穿透式过孔要穿透整个电路板，与不同的导电图形层建立电气连接关系，因此系统专门设置了一个抽象的层—多层。一般，焊盘与过孔都要设置在多层上，如果关闭此层，焊盘与过孔就无法显示出来。

![MultiLayer](https://raw.githubusercontent.com/eternityqjl/blogGallery/master/blog/MultiLayer.PNG)

### Drill layer(钻孔层) 

钻孔层提供电路板制造过程中的钻孔信息(如焊盘，过孔就需要钻孔)。Altium Designer提供了Drill gride(钻孔指示)和Drill drawing(钻孔)两个钻孔层。

## 原理图库器件画法

放置矩形

添加Pin

设置器件的Designator（例如P?）

## 器件的PCB封装画法

确定是过孔还是焊盘，然后根据坐标放置位置，最后画丝印。



## 批量添加封装并给各种原件命名

### 添加封装

右键-Find Similar Objects选择相同类型的原件-右键-Properties-修改封装

### 批量命名

执行**Tools-Annotation-Annotate Schematic**(或快捷键**T A A**)命令打开原理图元件命名管理器

左上角设置命名顺序

设置命名范围

然后单击Update

可以看到新的命名建议，然后执行有效角的接收更改

依次单击Validate和Execute（保证两排绿√）

即可完成命名

## 确定PCB范围并布局

根据需要的形状和经验粗略地确定版图的位置边界，然后布局元件位置。

## 布线

注意设置焊盘间的距离，jlc为7mil，以及过孔孔径。

过孔打法：`Alt+Shift+鼠标滚轮向上`

## 铺铜

快捷键`P+G`，选择铺铜类型，以及铺铜连接的NET。

## 其他：一些快捷键

* `ctrl+D`: View Configuration，用来配置PCB的显示内容