---
title: 第一次使用LaTex——问题整理
date: 2020-03-29 00:14:06
tags: LaTex使用中的问题
categories: LaTex
---

## 0.前言

第一次使用Tex语言完整地完成了一篇实验报告的构建排版，这过程中遇到了各种各样的问题，有些问题虽然解决了，但在原理方面还是一知半解，之后再遇到还希望能多看文档。这篇博客是来记录第一次编写过程中遇到的一些问题。

这是第一次实验报告导出的[pdf文档](http://blog.eternityqjl.top/SS_test1.pdf)封面展示：

![](http://blog.eternityqjl.top/SS_test1_cover.jpg)

这是[tex源代码](http://blog.eternityqjl.top/SS_test1.tex)。

<!-- more -->

## 章节标题的特殊符号

当我在章节标题的大括号内输入行内公式时出现了错误，查找后发现pdf书签中不能使用特殊符号，需要用命令`\texorpdfstring{}{}`分别定义在TEX文档输出的代码和PDF书签处的纯文本形式，第一个括号`{}`中填写tex文档代码，第二个括号`{}`填写文本形式的书签，如下所示：

```tex
%1.1
\subsection[第3题（1）]{第3题（1）求\texorpdfstring{$x_1(t)=1.5e^{-2t}u(t)$}{x1}，
\texorpdfstring{$x_2(t)=cos(t2)[u(t)-u(t-2)]$}{x2}的卷积积分
\texorpdfstring{$y_1(t)=x_1(t)*x_2(t)$}{y1}。 }
```

## 代码段的设置

在文章中添加代码段需要用到`listings`宏包，代码段的基本设置参数如下所示：

> [listings宏包文档](http://texdoc.net/texmf-dist/doc/latex/listings/listings.pdf)

```tex
\usepackage{listings}
\lstset{
language=Matlab,
escapeinside=``, 
numbers=left,
numberstyle=\tiny,
breaklines=true, 
backgroundcolor=\color{lightgray!40!white},
frame=single,
framerule=0pt,
extendedchars=false, 
keywordstyle=\color{blue!70}\bfseries, 
basicstyle=\ttfamily,
commentstyle=\ttfamily\color{green!40!black}, 
showstringspaces=false}
```

同时，设置代码段颜色高亮时需要用到`xcolor`宏包。

注意以下几点问题：

- 代码段中最好不要出现中文，如果必须要使用中文，需要用逃逸字符\` \`将中文包括起来，如上面的基本参数中的escapeinside=\` \` 所示。
- 如果代码段中出现tex语言的特殊符号，需要使用转义字符`\`，如`#`就需要转换为`\#`。

如果想要改变代码段中的字体，需要用到`fontspec`宏包，这个宏包需要用`XeLaTex`或`LuaLaTex`进行编译，由于我的实验报告封面使用的有些元素在使用这两者编译时无法识别，我这次就没使用，下次写实验报告一定仔细研究一下，实现这个功能。

## 数学公式中的align

> 参考：[“Erroneous nesting of equation structures” message- I can't see what's wrong?](https://tex.stackexchange.com/questions/372384/erroneous-nesting-of-equation-structures-message-i-cant-see-whats-wrong)

数学公式中需要写等号对齐的公式段是不需要写`$$...$$`或者`begin{equation}...end{equation}`，直接将公式写在`begin{align}...end{align}`的中间即可，如下所示：

```tex
\begin{align}
y_2[n]
&=x_3[n]*x_4[n] \\
&=((-\frac{2}{3})^n u[n-1])*((-1)^{n+1}u[n-1]-(-2)^{n-2}u[n-2]) \\
&=\sum_{k=-\infty}^{+\infty}(-\frac{2}{3})^k u[k-1]((-1)^{n-k+1}u[n-k+1]-(-2)^{n-k-2}u[n-k-2]) \\
&=\sum_{k=1}^{+\infty}(-\frac{2}{3})^k ((-1)^{n-k+1}u[n-k+1]-(-2)^{n-k-2}u[n-k-2]) \\
&=
\begin{cases}
	0, \ n<0 \\
	\sum_{k=1}^{+\infty}(-\frac{2}{3})^k (-1)^{n-k+1}u[n-k+1], \ 0 \leq n \leq 2 \\
	\sum_{k=1}^{+\infty}(-\frac{2}{3})^k ((-1)^{n-k+1}u[n-k+1]-(-2)^{n-k-2}u[n-k-2]),\ n>2
\end{cases}
\end{align}
```

![](http://blog.eternityqjl.top/equationSample.JPG)

