import pandas as pd
import re
import json

def process_answer(answer):
    """将答案处理成分步解答格式"""
    # 按句号分割
    steps = [step.strip() for step in answer.split('。') if step.strip()]
    
    # 如果步骤太少，尝试按其他分隔符分割
    if len(steps) < 2:
        steps = [step.strip() for step in answer.split('；') if step.strip()]
    if len(steps) < 2:
        steps = [step.strip() for step in answer.split('，') if step.strip()]
    
    # 如果仍然只有一个步骤，尝试按换行符分割
    if len(steps) < 2:
        steps = [step.strip() for step in answer.split('\n') if step.strip()]
    
    return steps

def identify_question_type(question):
    """识别问题类型"""
    math_patterns = {
        'algebra': r'方程|函数|代数|不等式',
        'geometry': r'几何|三角形|圆|面积|体积',
        'calculus': r'导数|积分|极限|微分',
        'probability': r'概率|统计|期望|方差'
    }
    
    programming_patterns = {
        'algorithm': r'算法|排序|查找|递归',
        'data_structure': r'数组|链表|树|图|栈|队列',
        'syntax': r'语法|错误|bug|调试'
    }
    
    for category, patterns in math_patterns.items():
        if re.search(patterns, question):
            return 'math', category
            
    for category, patterns in programming_patterns.items():
        if re.search(patterns, question):
            return 'programming', category
            
    return 'general', 'unknown'

def process_data():
    """处理原始数据并生成新的格式"""
    try:
        # 读取原始数据
        data = pd.read_csv('qa_data', 
                          sep='\t', 
                          names=['question', 'answer'],
                          encoding='utf-8',
                          on_bad_lines='skip')
        
        # 处理数据
        processed_data = []
        for _, row in data.iterrows():
            question = row['question']
            answer = row['answer']
            
            # 识别问题类型
            q_type, category = identify_question_type(question)
            
            # 处理答案
            steps = process_answer(answer)
            
            # 创建新的数据条目
            processed_entry = {
                'question': question,
                'type': q_type,
                'category': category,
                'steps': steps,
                'original_answer': answer
            }
            
            processed_data.append(processed_entry)
        
        # 保存处理后的数据
        with open('processed_qa_data.json', 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
        print(f"成功处理 {len(processed_data)} 条数据")
        
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")

if __name__ == "__main__":
    process_data() 