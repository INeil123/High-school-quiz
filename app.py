import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import json
from sklearn.metrics.pairwise import cosine_similarity
from model_handler import ModelHandler

# 设置页面配置
st.set_page_config(
    page_title="智能解题助手",
    page_icon="🤖",
    layout="wide"
)

# 加载模型
@st.cache_resource
def load_model():
    return ModelHandler()

# 加载数据
@st.cache_data
def load_data():
    try:
        with open('processed_qa_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"加载数据时出错: {str(e)}")
        return []

# 初始化会话状态
if 'model' not in st.session_state:
    st.session_state.model = load_model()
if 'data' not in st.session_state:
    st.session_state.data = load_data()
    if st.session_state.data:
        st.session_state.embeddings = st.session_state.model.retrieval_model.encode(
            [item['question'] for item in st.session_state.data]
        )
    else:
        st.session_state.embeddings = []

# 自定义CSS样式
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border: none;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .step-box {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .type-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 14px;
        margin-right: 10px;
    }
    .math-badge {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .programming-badge {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
    .general-badge {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .context-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# 标题
st.title("🤖 智能解题助手")
st.write(f"已加载数据条数：{len(st.session_state.data)}")
st.markdown("---")

# 用户输入
user_question = st.text_input("请输入您的问题：", placeholder="例如：如何求解二次方程？")

if user_question:
    with st.spinner("正在思考中..."):
        # 生成解答
        result = st.session_state.model.generate_solution(user_question)
        
        # 显示解答过程
        st.markdown("### 解答过程：")
        
        # 显示分步解答
        for i, step in enumerate(result['steps'], 1):
            st.markdown(f"""
            <div class="step-box">
                <strong>步骤 {i}:</strong> {step}
            </div>
            """, unsafe_allow_html=True)
        
        # 显示参考解答
        with st.expander("查看参考解答"):
            st.markdown("""
            <div class="context-box">
                {context}
            </div>
            """.format(context=result['context'].replace('\n', '<br>')), unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown("### 使用说明")
st.markdown("""
1. 在输入框中输入您的问题
2. 系统会自动分析问题并提供详细的解题步骤
3. 您可以查看参考解答以了解更多信息
4. 支持数学题、编程题等多种类型的问题
""") 