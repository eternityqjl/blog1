---
title: Dijkstra双栈——算术表达式求值
date: 2020-07-21 20:15:08
tags:
categories: 数据结构与算法
---

## 算数表达式

这里的算术表达式支持常见的二元运算符`+-*/`以及接受一个参数的平方根运算符`sqrt`。这里我们假定表达式中未省略所有的括号。

## 计算方法

* 将操作数压入**操作数栈**
* 将运算符压入**运算符栈**
* 忽略左括号
* 遇到右括号时，弹出一个运算符，弹出需要数量的操作数进行运算，然后将得到的结果再压入操作数栈。

## 代码实现

```java
package edu.princeton.cs.algs4;

public class Evaluate {
	public static void main(String[] args)
	{
		Stack<String> ops = new Stack<String>();
		Stack<Double> vals = new Stack<Double>();
		while (!StdIn.isEmpty())
		{
			//读取字符，如果是运算符则压入运算符栈ops
			String s = StdIn.readString();
			if (s.equals("("));
			else if (s.equals("+"))    ops.push(s);
			else if (s.equals("-"))    ops.push(s);
			else if (s.equals("*"))    ops.push(s);
			else if (s.equals("/"))    ops.push(s);
			else if (s.equals("sqrt"))    ops.push(s);
			
			//如果字符为)，则弹出运算符和操作数，计算结果并压入操作数栈vals
			else if (s.equals(")"))
			{
				String op = ops.pop();
				double v = vals.pop();
				if (op.equals("+"))    v = vals.pop() + v;
				else if (op.equals("-"))    v = vals.pop() - v;
				else if (op.equals("*"))    v = vals.pop() * v;
				else if (op.equals("/"))    v = vals.pop() / v;
				else if (op.equals("sqrt"))    v = Math.sqrt(v);
				
				vals.push(v);
			}
			//如果字符既非运算符又非括号，将其作为double值压入操作数栈vals
			else vals.push(Double.parseDouble(s));
		}
		StdOut.println(vals.pop());
	}
}
```

