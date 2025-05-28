# 高中数学知识问答系统

这是一个基于高中数学数据集的知识问答系统，使用Streamlit构建的Web界面。

## 功能特点

- 支持自然语言问题输入
- 使用语义相似度匹配最相关的答案
- 提供多个相关答案供参考
- 美观的用户界面
- 支持中文问答

## 安装说明

1. 确保已安装Python 3.8或更高版本
2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 运行方法

在命令行中执行：
```bash
streamlit run app.py
```

## 使用说明

1. 在浏览器中打开显示的地址（默认为 http://localhost:8501）
2. 在输入框中输入您的问题
3. 系统会自动搜索并显示最相关的答案
4. 您还可以查看其他相关的答案

## 技术栈

- Streamlit：Web界面框架
- Sentence-Transformers：语义相似度计算
- Pandas：数据处理
- Scikit-learn：相似度计算 