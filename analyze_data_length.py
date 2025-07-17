import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# 加载tokenizer
print("加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('/data_local/lww/verl/base_models/Qwen2.5-7B')

# 读取数据
print("读取数据...")
df = pd.read_parquet('train_test_data/simplelr_math_35/train.parquet')

# 计算prompt的token长度
print('正在计算prompt token长度...')
prompt_token_lens = []
for i, prompt in enumerate(df['prompt'][:1000]):  # 先分析前1000个样本
    tokens = tokenizer.encode(str(prompt))
    prompt_token_lens.append(len(tokens))
    if i % 200 == 0:
        print(f'已处理 {i+1} 个样本')

prompt_token_lens = np.array(prompt_token_lens)

print(f'\nPrompt Token长度统计 (前1000个样本):')
print(f'最小值: {prompt_token_lens.min()}')
print(f'最大值: {prompt_token_lens.max()}')
print(f'平均值: {prompt_token_lens.mean():.1f}')
print(f'中位数: {np.median(prompt_token_lens):.1f}')
print(f'75分位数: {np.percentile(prompt_token_lens, 75):.1f}')
print(f'90分位数: {np.percentile(prompt_token_lens, 90):.1f}')
print(f'95分位数: {np.percentile(prompt_token_lens, 95):.1f}')

# 分析prompt截断影响
print(f'\nPrompt截断影响分析:')
print(f'超过2048 tokens的样本: {(prompt_token_lens > 2048).sum()} / {len(prompt_token_lens)} ({(prompt_token_lens > 2048).mean()*100:.1f}%)')
print(f'超过1024 tokens的样本: {(prompt_token_lens > 1024).sum()} / {len(prompt_token_lens)} ({(prompt_token_lens > 1024).mean()*100:.1f}%)')
print(f'超过512 tokens的样本: {(prompt_token_lens > 512).sum()} / {len(prompt_token_lens)} ({(prompt_token_lens > 512).mean()*100:.1f}%)')

# 分析response长度 - 基于现有的answer和gt_answer
print('\n正在分析response长度...')
answer_token_lens = []
gt_answer_token_lens = []

for i in range(min(1000, len(df))):
    # 分析answer列
    answer_tokens = tokenizer.encode(str(df.iloc[i]['answer']))
    answer_token_lens.append(len(answer_tokens))
    
    # 分析gt_answer列  
    gt_answer_tokens = tokenizer.encode(str(df.iloc[i]['gt_answer']))
    gt_answer_token_lens.append(len(gt_answer_tokens))
    
    if i % 200 == 0:
        print(f'已处理response {i+1} 个样本')

answer_token_lens = np.array(answer_token_lens)
gt_answer_token_lens = np.array(gt_answer_token_lens)

print(f'\nAnswer Token长度统计:')
print(f'最小值: {answer_token_lens.min()}')
print(f'最大值: {answer_token_lens.max()}')
print(f'平均值: {answer_token_lens.mean():.1f}')
print(f'中位数: {np.median(answer_token_lens):.1f}')
print(f'75分位数: {np.percentile(answer_token_lens, 75):.1f}')
print(f'90分位数: {np.percentile(answer_token_lens, 90):.1f}')
print(f'95分位数: {np.percentile(answer_token_lens, 95):.1f}')

print(f'\nGT Answer Token长度统计:')
print(f'最小值: {gt_answer_token_lens.min()}')
print(f'最大值: {gt_answer_token_lens.max()}')
print(f'平均值: {gt_answer_token_lens.mean():.1f}')
print(f'中位数: {np.median(gt_answer_token_lens):.1f}')
print(f'75分位数: {np.percentile(gt_answer_token_lens, 75):.1f}')
print(f'90分位数: {np.percentile(gt_answer_token_lens, 90):.1f}')
print(f'95分位数: {np.percentile(gt_answer_token_lens, 95):.1f}')

# 模拟实际训练中的response长度（数学问题的详细解答）
print('\n模拟训练中response长度分析:')
print('注意：训练时模型会生成详细的解题过程，通常比简短答案长得多')

# 基于数学问题的特点，估算实际response长度
# 数学问题通常需要：问题理解 + 解题步骤 + 计算过程 + 最终答案
estimated_response_lens = []
for i in range(min(100, len(df))):
    # 根据prompt长度和问题复杂度估算response长度
    prompt_len = prompt_token_lens[i]
    # 数学问题的response通常是prompt长度的2-5倍
    estimated_len = prompt_len * np.random.uniform(2, 5)
    estimated_response_lens.append(int(estimated_len))

estimated_response_lens = np.array(estimated_response_lens)

print(f'\n估算的训练Response长度统计 (基于前100个样本):')
print(f'最小值: {estimated_response_lens.min()}')
print(f'最大值: {estimated_response_lens.max()}')
print(f'平均值: {estimated_response_lens.mean():.1f}')
print(f'中位数: {np.median(estimated_response_lens):.1f}')
print(f'75分位数: {np.percentile(estimated_response_lens, 75):.1f}')
print(f'90分位数: {np.percentile(estimated_response_lens, 90):.1f}')
print(f'95分位数: {np.percentile(estimated_response_lens, 95):.1f}')

print(f'\nResponse截断影响分析 (基于估算):')
print(f'超过2048 tokens的样本: {(estimated_response_lens > 2048).sum()} / {len(estimated_response_lens)} ({(estimated_response_lens > 2048).mean()*100:.1f}%)')
print(f'超过1024 tokens的样本: {(estimated_response_lens > 1024).sum()} / {len(estimated_response_lens)} ({(estimated_response_lens > 1024).mean()*100:.1f}%)')
print(f'超过512 tokens的样本: {(estimated_response_lens > 512).sum()} / {len(estimated_response_lens)} ({(estimated_response_lens > 512).mean()*100:.1f}%)')

# 总序列长度分析
total_lens = prompt_token_lens[:100] + estimated_response_lens
print(f'\n总序列长度 (Prompt + Response) 统计:')
print(f'最小值: {total_lens.min()}')
print(f'最大值: {total_lens.max()}')
print(f'平均值: {total_lens.mean():.1f}')
print(f'中位数: {np.median(total_lens):.1f}')
print(f'75分位数: {np.percentile(total_lens, 75):.1f}')
print(f'90分位数: {np.percentile(total_lens, 90):.1f}')
print(f'95分位数: {np.percentile(total_lens, 95):.1f}')

print(f'\n总序列长度截断影响分析:')
print(f'超过4096 tokens的样本: {(total_lens > 4096).sum()} / {len(total_lens)} ({(total_lens > 4096).mean()*100:.1f}%)')
print(f'超过2048 tokens的样本: {(total_lens > 2048).sum()} / {len(total_lens)} ({(total_lens > 2048).mean()*100:.1f}%)')
print(f'超过1024 tokens的样本: {(total_lens > 1024).sum()} / {len(total_lens)} ({(total_lens > 1024).mean()*100:.1f}%)')

# 显示几个样本
print(f'\n样本示例:')
for i in range(3):
    print(f'样本 {i+1}:')
    print(f'  Prompt Token长度: {prompt_token_lens[i]}')
    print(f'  Answer Token长度: {answer_token_lens[i]}')
    print(f'  估算Response长度: {estimated_response_lens[i] if i < len(estimated_response_lens) else "N/A"}')
    print(f'  Answer内容: {df.iloc[i]["answer"]}')
    print(f'  Prompt预览: {str(df.iloc[i]["prompt"])[:150]}...')
    print() 