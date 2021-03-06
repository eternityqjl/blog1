---
title: TLS（HTTPS）协议
date: 2020-04-19 16:15:26
tags: TLS/SSL协议
categories: 计算机网络学习日志
---

## 1. SSL/TLS简介

**TLS**（**T**ransport **L**ayer **S**ecurity，**传输层安全协议**），以及它的前身**SSL**（**S**ecure **S**ockets **L**ayer，**安全套接层**）是一种安全协议。Netscape公司在1994年推出HTTPS协议，使用SSL进行加密，这是SSL的起源。1999年公布了第一版的TLS协议，类似于SLLv3，只是对其做出了一些更改。

SSL协议处在计算机网络中的应用层和运输层之间，它不依赖于平台和运行程序的协议。

**几个知名的使用SSL加密的协议**：

**HTTP over SSL (HTTPS)**

简称为HTTPS，它的产生是为了加密网页，HTTP是第一个使用SSL保障安全的应用层协议。HTTPS在RFC2818被标准化，HTTPS工作在443端口，HTTP默认工作在80端口。

**Email over SSL**

类似于HTTP over SSL，协议如下：

* SMTP、IMAP能够支持SSL
* SMTP over TLS在标准文档RFC2487中

<!-- more -->

## 2. SSL原理详解

### 2.0 基本运行过程

SSL/TLS协议是采用**公钥加密法**实现的，客户端先向服务器端索要公钥，用公钥加密信息，服务器收到密文后，用自己的私钥解密；二者协商生成”对话密钥“，采用该”对话密钥“进行加密通信。

### 2.1 SSL建立的总过程

客户端向服务器索要并验证公钥，双方协商生成”对话密钥“的过程又称为”握手阶段“，该阶段涉及**四次握手**通信，且该阶段的通信都是明文的。以下一一来分析。

#### 1. 客户端发出请求（ClientHello）

该步客户端（通常为浏览器）向服务器提供以下信息：

* 支持的协议**版本（Version）**，如TLSv1.0
* 一个客户端生成的**随机数（Random）**，之后用于生成”对话密钥“
* **会话ID（Session id）**：
  * 如果客户端第一次连接到服务器，那么该字段为空。
  * 如果该字段不为空，则说明以前与服务器有连接，在此期间，服务器使用Session ID映射对称密钥，并将Session ID存储在客户端浏览器中，为映射设置一个时间限，如果浏览器将来连接到同一台服务器，它将发送Session ID，服务器对映射的Session ID进行验证，并使用以前用过的对称密钥来恢复会话，该情况下不需要握手，也成为**SSL会话恢复**。
* 支持的**加密套件（Cipher Suites）**，这是由客户按优先级排列的，但完全由服务器决定发送与否。服务器会从客户端发送的加密套件中选择一种作为共同的加密套件，如RSA公钥加密。
* 支持的**压缩方法**，这是为了减少带宽。从TLS 1.3开始，协议禁用了TLS压缩，因为使用压缩时攻击可以捕获到用HTTP头发送的参数，该攻击可以劫持Cookie。
* 扩展包。

#### 2. 服务器回应（ServerHello）

该步骤包含以下内容：

* 确认使用的**版本**，如TLSv1.0，如果浏览器与服务器支持的版本不一致，服务器会关闭加密通信。
* 一个服务器生成的**随机数**，之后用于生成”对话密钥“
* 确认使用的**加密套件**
* **会话ID**（Session ID）：
  * 服务器将约定的Session参数存储在TLS缓存中，并生成与之对应的Session ID，它将与ServerHello一起发送到客户端。客户端可以写入约定的参数到此Session ID，并给定到期时间，客户端将在ClientHello中包含该ID。如果客户端再次连接到该服务器，服务器可以检查与Session ID对应的缓存参数，并重用他们而无需再次握手。这可以节省大量计算成本。
  * 但在谷歌等大流量应用程序中这种方法存在缺点，每天有数百万人连接到服务器，服务器必须使用Session ID保留所有Session参数的TLS缓存，这是一笔巨大的开销。为解决该问题，在扩展包中加入**Session Tickets**，在这里客户端可以在ClientHello中指定它是否支持Session Tickets，然后服务器将创建一个新的Session Tickets，并使用只有服务器知道的经过私钥加密的Session参数，该参数存储在客户端中，因此所有Session数据仅存储在客户端计算机上，但Ticket仍然是安全的，因为该密钥只有服务器知道。
* **扩展包**
* **服务器证书**

当服务器需要确认客户端身份时，就会再包含一个请求，要求客户端提供”客户端证书“。例如金融机构只允许认证客户连入自己的网络，回向正式用户提供USB密钥（U盾），里面包含一张客户端证书。

#### 3. 客户端回应

客户端收到服务器回应后首先验证服务器的证书，如果证书存在问题，如证书过期、由非可信机构颁布、或证书域名与实际域名不一致，会想客户端访问者发出警告，询问是否继续通信。

证书没有问题则客户端会从中取出公钥然后发送以下信息：

* 一个**随机数（pre-master-key)**；该随机数用服务器公钥加密，防止被窃听。
* 编码改变通知，表示随后的信息都将用双方商定的加密方法和密钥发送。
* 客户端结束握手通知，该项同时也是前面发送所有内容的哈希值，用来供服务器验证。

上面第一项的随机数是握手阶段出现的第三个随机数，称“pre-master-key”，之后客户端和服务器就同时有了3个随机数，接着用双方事先商定的加密方法各自生成本次会话用的同一把“会话密钥”。

> pre-master-key与前面服务器和客户端在Hello阶段产生的两个随机数结合在一起生成了Master Secret。

#### 4. 服务器的最后回应

服务器收到客户端第三个随机数pre-master-key后，计算生成本次会话使用的“会话密钥”，然后向客户端发送以下信息：

* 编码改变通知，表示随后的信息都将用双方商定的加密方法和密钥发送。
* 服务器握手结束通知，该项同时也是前面发送的所有内容的哈希值，用来供客户端验证。

至此整个握手阶段就结束了，接下来客户端与服务器进入加密通信，就是完全使用普通的HTTP协议，只是使用了“会话密钥”加密内容。

### 2.2 SSL协议的结构体系

![](http://blog.eternityqjl.top/SSL%E5%8D%8F%E8%AE%AE%E4%BD%93%E7%B3%BB%E7%BB%93%E6%9E%84_1.png)

SSL体系结构包含两个协议子层，底层为**SSL记录协议层**；高层为**SSL握手协议层**。

* **SSL记录协议层**：记录协议为高层协议提供基本的安全服务，如数据封装、压缩、加密等基本功能。所以我们可以知道，所有的传输数据都被封装在记录中。
* **SSL握手协议层**包括：
  * SSL握手协议：协调客户和服务器的状态，是双方能达到状态的同步
  * SSL密码参数修改协议：更新用于当前连接的密码组。
  * SSL告警协议：发现异常时为对等方传递警告

## 3. 参考

* [SSL/TLS协议详解|曹世宏的博客](https://cshihong.github.io/2019/05/09/SSL协议详解/)
* [SSL/TLS协议运行机制的概述|阮一峰](https://www.ruanyifeng.com/blog/2014/02/ssl_tls.html)