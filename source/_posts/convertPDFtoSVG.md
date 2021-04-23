---
title: 将PDF文档中的矢量图导出为SVG格式的图片
date: 2020-04-08 00:15:48
tags:
---

## 前言

最近在写笔记文档和实验报告的时候遇到一些处理资料图片的问题，PDF电子书或文档中的插图一般为矢量图，自己写笔记如果需要引用这些插图时，一般会直接截图，截图为JEPG或PNG格式，并非矢量图，会在新的文档中出现失真。所以为了解决这个问题，我使用**Adobe Acrobat**和**inkscape**两个软件将PDF中的矢量图导出为**SVG**格式的矢量图。以下为具体步骤：

## 在`Adobe Acrobat`中选择`编辑PDF`选项

![](http://blog.eternityqjl.top/convertingPDFtoSVG_1.JPG)

<!-- more -->

## 选择`裁剪页面`

![](http://blog.eternityqjl.top/convertingPDFtoSVG_2.JPG)

## 框选所需图片的区域

![](http://blog.eternityqjl.top/convertingPDFtoSVG_3.JPG)

## 双击框选的区域，按如下所示设置

![](http://blog.eternityqjl.top/convertingPDFtoSVG_4.JPG)

## 选择`组织页面`

![](http://blog.eternityqjl.top/convertingPDFtoSVG_5.JPG)

## 选定裁切后的页面![](http://blog.eternityqjl.top/convertingPDFtoSVG_6.JPG)

## 右击后选择`打印页面`![](http://blog.eternityqjl.top/convertingPDFtoSVG_7.png)

## 设置页面：按照PDF页面大小选择纸张来源，这样选择后最终导出的图片范围才是裁切时所选范围。然后打印。

![](http://blog.eternityqjl.top/convertingPDFtoSVG_8.JPG)

## 用`inkscape`打开刚才打印的PDF文件，注意，导入时设置为第一项`从Poppler/Cairo导入`，类似于PS中的栅格化，这样才能保证图中的字体不发生改变

![](http://blog.eternityqjl.top/convertingPDFtoSVG_9.JPG)

## 选择`文件`$\rightarrow$`文档属性`，设置自定义尺寸为`缩放页面到内容`$\rightarrow$`缩放页面到绘图或选区`

![](http://blog.eternityqjl.top/convertingPDFtoSVG_10_0.png)

![](http://blog.eternityqjl.top/convertingPDFtoSVG_10.JPG)

![](http://blog.eternityqjl.top/convertingPDFtoSVG_11.png)

## `文件`$\rightarrow$`另存为`，选择自己所需的格式，我一般选`SVG`

![](http://blog.eternityqjl.top/convertingPDFtoSVG_12.png)

![](http://blog.eternityqjl.top/convertingPDFtoSVG_13.JPG)