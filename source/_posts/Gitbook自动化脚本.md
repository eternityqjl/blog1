---
title: GitBook自动发布脚本
date: 2019-10-18 13:50:49
tags: GitBook
categories: 其他
---

GitBook作为一个非常好用的基于Node.js的命令行工具，可以输出HTML、PDF、eBook等多种格式的电子书，这里就不再详细讲述GitBook的构建以及发布到Github托管的过程，网络上有非常多的教程，可以自行Google查看。

刚开始在每次更新内容的时候都需要一系列的git命令将原始内容仓库和构建的网页仓库同步到Github上，大概需要10条命令，如果你的更新频率很高的话，发布内容将会是有个痛苦的过程，但这里我们可以通过一个shell脚本只使用`$ sh deploy.sh`一条命令完成同步内容和构建网页的所有过程。

<!-- more -->

首先，在你的GitBook内容根目录下建一个`deploy.sh`文件，使用文本编辑器打开，然后输入以下内容：

```sh
#!bin/sh
git checkout master
# 切换到master分支，及内容所在的仓库
git add .
git commit -m "Update"
git push -u origin master
# 添加、提交到Git仓库，然后push到Github上
gitbook build
# 构建Gitbook

git checkout gh-pages
# 切换到gh-pages分支，即生成的HTML网页的仓库
cp -r _book/* . 
# 复制前面构建的内容
git add .
git commit -m "Update"
git push -u origin gh-pages
# 添加、提交到Git仓库，然后push到Github上
git checkout master
# 返回master主分支
```

然后打开命令行，为这个脚本授权：

```sh
$ chmod +x deploy.sh
```

至此就完成了，以后在更新的时候只需要输入一条命令`$ sh deploy.sh`就可以完成所有操作了。

