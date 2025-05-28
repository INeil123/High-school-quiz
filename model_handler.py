import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import json
import re
import os

# 设置 HF-Mirror 环境变量（可选，保留以防后续需要）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class ModelHandler:
    def __init__(self):
        # 初始化检索模型
        self.retrieval_model = SentenceTransformer('./model_cache/')
        
        # 从本地 model_cache 文件夹加载模型
        self.tokenizer = AutoTokenizer.from_pretrained("./model_cache", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "./model_cache",
            trust_remote_code=True
        ).eval()
        
        # 加载知识库数据
        with open('processed_qa_data.json', 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
            
        # 预计算知识库的嵌入向量
        self.kb_embeddings = self.retrieval_model.encode([item['question'] for item in self.knowledge_base])
    
    def get_relevant_context(self, question, top_k=3):
        """获取相关问题及其解答作为上下文"""
        question_embedding = self.retrieval_model.encode([question])
        similarities = torch.nn.functional.cosine_similarity(
            torch.tensor(question_embedding),
            torch.tensor(self.kb_embeddings)
        )[0]
        top_k = min(top_k, len(similarities))
        top_indices = torch.topk(similarities, top_k).indices.tolist()
        context = []
        
        for idx in top_indices:
            item = self.knowledge_base[idx]
            context.append(f"问题：{item['question']}\n解答：{item['original_answer']}\n")
            
        return "\n".join(context)
    
    def generate_solution(self, question):
        """生成解题思路和步骤"""
        # 获取相关上下文
        context = self.get_relevant_context(question)
        
        # 构建提示模板
        prompt = f"""请根据以下参考解答，为这个问题提供详细的解题思路和步骤：

参考解答：
{context}

问题：{question}

请按照以下格式提供解答：
1. 分析问题类型和关键点
2. 列出解题步骤
3. 给出最终答案

解答："""
        
        # 生成解答
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的解答部分
        solution = response.split("解答：")[-1].strip()
        
        # 将解答分解为步骤
        steps = self._parse_solution_steps(solution)
        
        return {
            'solution': solution,
            'steps': steps,
            'context': context
        }
    
    def _parse_solution_steps(self, solution):
        """将解答文本解析为步骤列表"""
        # 尝试按数字编号分割
        steps = re.split(r'\d+[.、]', solution)
        steps = [step.strip() for step in steps if step.strip()]
        
        # 如果没有找到数字编号，尝试按其他分隔符分割
        if len(steps) < 2:
            steps = re.split(r'[。；]', solution)
            steps = [step.strip() for step in steps if step.strip()]
        
        # 如果仍然只有一个步骤，尝试按换行符分割
        if len(steps) < 2:
            steps = solution.split('\n')
            steps = [step.strip() for step in steps if step.strip()]
        
        return steps 