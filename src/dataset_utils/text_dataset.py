from multiprocessing.spawn import prepare
import os
import json, chardet
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Value, DatasetDict
from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase, default_data_collator

from dataset_utils.denoising_collator import DataCollatorForBartDenoisingLM
from dataset_utils.flan_collator import DataCollatorForFlanLM

import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import torch

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def pad_matrix(mat, max_node):
    """Pad or truncate matrix to [max_node, max_node]"""
    N = mat.shape[0]
    # 创建一个全零的输出矩阵
    out = np.zeros((max_node, max_node), dtype=np.float32)

    if N > max_node:
        # 如果矩阵大于 max_node，裁剪矩阵
        out = mat[:max_node, :max_node]
    elif N < max_node:
        # 如果矩阵小于 max_node，填充矩阵
        out[:N, :N] = mat
    else:
        # 如果矩阵的大小恰好等于 max_node，直接返回原矩阵
        out = mat

    return out


def exists(x):
    return x is not None


def get_code_dataset(dataset_name, metadata=False, synthetic_train_path=None):
    if dataset_name == 'cq_code':
        cq_code_data_path = 'datasets/cq_code/code_data.xlsx'
        #dataset = process_only_cqcode_dataset(cq_code_data_path)
    elif dataset_name == 'python_code':
        code_data_path = 'datasets/code_search_net/python/'
        dataset = process_code_search_net_dataset(code_data_path)
    elif dataset_name == 'multiple_code':
        code_data_path = 'datasets/code_search_net/multiple/'
        dataset = process_code_search_net_dataset(code_data_path)

    else:
        raise NotImplementedError
    return dataset

def get_cq_dataset(dataset_name, metadata=False, synthetic_train_path=None):
    if dataset_name == 'cq_code':
        cq_code_data_path = 'datasets/cq_code/split_data/'
        dataset = process_cq_dataset(cq_code_data_path)
    else:
        raise NotImplementedError
    return dataset


def get_dataset(dataset_name, metadata=False, synthetic_train_path=None):
    if dataset_name == 'roc':
        roc_data_path = 'datasets/ROCstory'
        dataset = load_dataset("text", data_files={f'{split}': os.path.join(roc_data_path, f'roc_{split}.json') for split in ['train', 'valid']})
        dataset = process_roc_dataset(dataset)
    elif dataset_name == 'ag_news':
        dataset = load_dataset('pietrolesci/ag_news', 'original')
        train_ds = dataset['train']
        train_val_ds = train_ds.train_test_split(test_size=1000, seed=42)
        train_val_ds['valid'] = train_val_ds['test']
        train_val_ds['test'] = dataset['test']
        dataset = process_ag_news_dataset(train_val_ds)
    elif dataset_name == 'xsum':
        dataset = load_dataset('xsum')
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_xsum_dataset(dataset)
    elif dataset_name == 'qqp':
        qqp_data_path = 'datasets/qqp'
        dataset = load_dataset("text", data_files={f'{split}': os.path.join(qqp_data_path, f'{split}.jsonl') for split in ['train', 'valid', 'test']})
        dataset = process_qqp_dataset(dataset)
    elif dataset_name == 'wmt14-de-en':
        #wmt14_de_en_data_path = 'D:\\Develop\\py_envs\\cache\\huggingface\\datasets\\wmt14\\de-en'
        dataset = load_dataset('wmt14', 'de-en')
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_wmt14_dataset(dataset, 'de-en')
    elif dataset_name == 'wmt14-de-de':
        dataset = load_dataset('wmt14', 'de-en')
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_wmt14_dataset(dataset, 'de-de')
    elif dataset_name == 'wmt14-en-de':
        dataset = load_dataset('wmt14', 'de-en')
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_wmt14_dataset(dataset, 'en-de')
    elif dataset_name == 'wmt14-en-en':
        dataset = load_dataset('wmt14', 'de-en')
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_wmt14_dataset(dataset, 'en-en')


    else:
        raise NotImplementedError
    return dataset


def process_roc_dataset(dataset):
    def extract_roc_text(example):
        text = example['text']
        assert text[:2] == '["'
        assert text[-2:] == '"]'
        sentences = text[2:-2]
        return {'text': sentences}
    dataset = dataset.map(extract_roc_text, )
    dataset = dataset.shuffle(seed=42)
    # Hold out some validation samples for testing
    val_test_ds = dataset['valid'].train_test_split(train_size=1000, shuffle=False)
    dataset['valid'] = val_test_ds['train']
    dataset['test'] = val_test_ds['test']
    return dataset

def process_ag_news_dataset(dataset):
    def process_ag_news_text(example):
        # return {'text': PreTrainedTokenizerBase.clean_up_tokenization(f'Title: {example["title"]}<pad> Description: {example["description"]}'.strip()), 'label':example['label']-1}
        return {'text': PreTrainedTokenizerBase.clean_up_tokenization(example["description"].strip()), 'label':example['label']-1}
    dataset = dataset.map(process_ag_news_text, remove_columns=['title', 'description', 'class'])
    return dataset

def process_xsum_dataset(dataset):
    def process_xsum_text(example):
        return {'text': PreTrainedTokenizerBase.clean_up_tokenization(example["summary"].strip()), 'context':PreTrainedTokenizerBase.clean_up_tokenization(example["document"].strip())}
    dataset = dataset.map(process_xsum_text, remove_columns=['summary', 'document', 'id'])
    dataset = dataset.shuffle(seed=42)
    return dataset

def process_qqp_dataset(dataset):
    def process_qqp_text(example):
        dict_example = json.loads(example['text'])
        dict_example['text'] = dict_example['trg']
        dict_example['context'] = dict_example['src']
        del dict_example['trg']
        del dict_example['src']
        return dict_example
    dataset = dataset.map(process_qqp_text, )
    dataset = dataset.shuffle(seed=42)
    return dataset


def process_only_cqcode_dataset(dataset, seed=43):

    #df = pd.read_excel(dataset)

    output_dir = os.path.dirname('datasets/cq_code/code_data/')  # 获取原文件目录
    # 4. 保存划分后的数据集为CSV文件
    train_path = os.path.join(output_dir, 'train_set.csv')
    val_path = os.path.join(output_dir, 'validation_set.csv')
    test_path = os.path.join(output_dir, 'test_set.csv')
    train_df = pd.read_csv(train_path, encoding='utf-8')
    valid_df = pd.read_csv(val_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8')

    from datasets import Dataset
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(
            train_df[['cq_code', 'nl']].rename(columns={'cq_code': 'text', 'nl': 'context'})),
        'valid': Dataset.from_pandas(valid_df[['cq_code', 'nl']].rename(columns={'cq_code': 'text', 'nl': 'context'})),
        'test': Dataset.from_pandas(test_df[['cq_code', 'nl']].rename(columns={'cq_code': 'text', 'nl': 'context'}))
    })

    return dataset_dict

def process_code_search_net_dataset(dataset, seed=43):

    #df = pd.read_excel(dataset)
    train_path = os.path.join(dataset, 'with_ast.jsonl')
    df = pd.read_json(train_path, lines=True)
    # 随机选取 300 个样本作为验证集（不放回抽样）
    train_df = df[df['lang'] == 'python'].sample(n=80000, random_state=seed)

    valid_df = train_df[train_df['lang'] == 'python'].sample(n=300, random_state=seed)

    # 剩余部分作为训练集
    train_df = train_df.drop(valid_df.index)  # 从原数据中移除验证集
    test_df = valid_df


    from datasets import Dataset
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(
            train_df[['code', 'text', 'ast']].rename(columns={'code': 'text', 'text': 'context'})),
        'valid': Dataset.from_pandas(valid_df[['code', 'text', 'ast']].rename(columns={'code': 'text', 'text': 'context'})),
        'test': Dataset.from_pandas(test_df[['code', 'text', 'ast']].rename(columns={'code': 'text', 'text': 'context'})),
    })

    return dataset_dict

def process_cq_dataset(dataset, seed=43):
    output_dir = dataset  # 获取原文件目录
    # 4. 保存划分后的数据集为CSV文件
    train_path = os.path.join(output_dir, 'train.jsonl')
    val_path = os.path.join(output_dir, 'val.jsonl')
    test_path = os.path.join(output_dir, 'test.jsonl')

    train_df = pd.read_json(train_path, lines=True)
    valid_df = pd.read_json(val_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    from datasets import Dataset
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df[['code', 'text']]),   #.rename(columns={'cq_code': 'text', 'nl': 'context'})
        'valid': Dataset.from_pandas(valid_df[['code', 'text']]),
        'test': Dataset.from_pandas(test_df[['code', 'text']])
    })

    return dataset_dict

def process_wmt14_dataset(dataset, lang_pair):
    def process_wmt14_text(example, lang_pair):
        source, target = lang_pair.split('-')
        assert source in ['de', 'en']
        assert target in ['de', 'en']

        return {'text': PreTrainedTokenizerBase.clean_up_tokenization(example['translation'][target].strip()), 'context':PreTrainedTokenizerBase.clean_up_tokenization(example['translation'][source].strip())}
    dataset = dataset.map(process_wmt14_text, fn_kwargs={'lang_pair': lang_pair}, remove_columns=['translation'])
    dataset = dataset.shuffle(seed=42)
    return dataset

def parse_metadata(metadata):
    if type(metadata) == list:
        return ' | '.join(metadata)
    elif type(metadata) == float:
        return 'Positive' if metadata > 0.5 else 'Negative'

def get_cq_dataloader(args, dataset, tokenizer, max_seq_len, mode='diffusion', shuffle=True, context_tokenizer=None):
    def tokenization(example):
        # print('EXAMPLE: ', example)
        assert context_tokenizer is not None
        source = example['text']
        target = example['code']
        cond_inputs = context_tokenizer(source, padding="max_length", truncation=True, max_length=max_seq_len)

        model_inputs = tokenizer(text_target=target, padding="max_length", truncation=True, max_length=max_seq_len)

        for k in cond_inputs.keys():
            model_inputs[f'cond_{k}'] = cond_inputs[k]

        return model_inputs


    if 'codet5' in args.enc_dec_model:
        collate_fn = DataCollatorForFlanLM(tokenizer)
    elif 't5' in args.enc_dec_model:
        collate_fn=DataCollatorForFlanLM(tokenizer)
    else:
        raise NotImplementedError

    dataset = dataset.map(tokenization, remove_columns=['text', 'code'], batched=True, num_proc=None)

    dl = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=4
    )
    return dl


"""
def get_dataloader(args, split, dataset, tokenizer, max_seq_len, mode='diffusion', shuffle=True, context_tokenizer=None):

    def tokenization(example):
        def process_ast_tokens(ast_tokens, tokenizer, max_ast_len, max_token_len):
            tokenized_list = []

            for token_str in ast_tokens:
                encoding = tokenizer(
                    token_str,
                    padding='max_length',
                    truncation=True,
                    max_length=max_token_len,
                    return_tensors='pt'
                )
                tokenized_list.append(encoding['input_ids'][0])  # shape: [max_token_len]

            # Padding 空节点为全0
            pad_length = max_ast_len - len(tokenized_list)
            if pad_length > 0:
                zero_tensor = torch.zeros(max_token_len, dtype=torch.long)
                tokenized_list.extend([zero_tensor.clone()] * pad_length)
            else:
                tokenized_list = tokenized_list[:max_ast_len]

            # 拼接成 [max_ast_len, max_token_len]
            token_id_tensor = torch.stack(tokenized_list, dim=0)
            return token_id_tensor  # LongTensor, shape: [max_ast_len, max_token_len]

        # print('EXAMPLE: ', example)
        if mode == 'diffusion' and args.dataset_name in {'xsum', 'qqp',  'wmt14-en-de', 'wmt14-de-en', 'cq_code', 'python_code', 'multiple_code'}:
            # import pdb; pdb.set_trace()
            assert context_tokenizer is not None
            source = example['context']
            target = example['text']
            ast = example['ast']
            if isinstance(ast, list) and isinstance(ast[0], str):
                ast = [json.loads(node) for node in ast]

            if args.dataset_name in {'qqp', 'wmt14-en-de', 'wmt14-de-en', 'cq_code', 'python_code', 'multiple_code'}:
                cond_inputs = context_tokenizer(source, padding="max_length", truncation=True, max_length=max_seq_len)
            elif args.dataset_name in {'xsum',}:
                cond_inputs = context_tokenizer(source, padding="max_length", truncation=True, max_length=max_seq_len*4)
            else:
                raise NotImplementedError

            model_inputs = tokenizer(text_target=target, padding="max_length", truncation=True, max_length=max_seq_len)

            model_inputs['labels'] = [x.copy() for x in model_inputs['input_ids']]

            # 1. 获取邻接边
            id2children = {node['id']: node['children'] for node in ast if 'children' in node}
            edges = [(parent, child) for parent, children in id2children.items() for child in children]

            # 2. 构建邻接矩阵 A
            max_node = args.max_ast_len
            num_nodes = len(ast)
            A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            for src, tgt in edges:
                if src < num_nodes and tgt < num_nodes:
                    A[src, tgt] = 1
                    A[tgt, src] = 1  # 如果使用无向图

            A = A + np.eye(num_nodes, dtype=np.float32)  # 加自环
            A = pad_matrix(A, max_node)
            A_norm = normalize(A)
            A_tensor = torch.FloatTensor(A_norm)

            # 3. 构建 A² ~ A⁵
            A_powers = []
            A_k = A.copy()
            for _ in range(2, 6):  # A², A³, A⁴, A⁵
                A_k = np.dot(A_k, A)
                A_powers.append(torch.FloatTensor(normalize(pad_matrix(A_k, max_node))))

            # 4. 提取 AST 节点的 token（不生成嵌入）
            ast_tokens = []
            for node in ast:
                node_type = node['type']
                node_val = node.get('value', '')
                if node_val:
                    ast_tokens.append(f"{node_type}_{node_val}")
                else:
                    ast_tokens.append(node_type)

            ### 存入 model_inputs 中
            model_inputs['ast_adj'] = A_tensor  # 一阶邻接矩阵（A）
            model_inputs['ast_adj_2'] = A_powers[0]  # A²
            model_inputs['ast_adj_3'] = A_powers[1]  # A³
            model_inputs['ast_adj_4'] = A_powers[2]  # A⁴
            model_inputs['ast_adj_5'] = A_powers[3]  # A⁵

            model_inputs['ast_token_ids'] = process_ast_tokens(
                ast_tokens=ast_tokens,
                tokenizer=tokenizer,
                max_ast_len=args.max_ast_node_num,
                max_token_len=args.max_token_len
            )

            # Add model target to model inputs
            for k in cond_inputs.keys():
                model_inputs[f'cond_{k}'] = cond_inputs[k]

            return model_inputs
        else:
            text = example["text"]
        return tokenizer(text, padding="max_length", truncation=True, max_length=max_seq_len)

    if 'mbart' in args.enc_dec_model:
        collate_fn=default_data_collator
    #elif 'bart' in args.enc_dec_model:
    #    collate_fn=DataCollatorForBartDenoisingLM(tokenizer, model_config.decoder_start_token_id)
    elif 'codet5' in args.enc_dec_model:
        collate_fn = DataCollatorForFlanLM(tokenizer)
    elif 't5' in args.enc_dec_model:
        collate_fn=DataCollatorForFlanLM(tokenizer)
    else:
        raise NotImplementedError
    
    if args.dataset_name in {'xsum', 'qqp'} or 'code' or 'cq_code' in args.dataset_name:
        dataset = dataset.map(tokenization, remove_columns=['text', 'context', 'ast'], batched=True, num_proc=None)
    else:
        dataset = dataset.map(tokenization, remove_columns='text')

    if split == 'train':
        batch_size = args.train_batch_size
    else:
        batch_size = args.eval_batch_size//2

    dl = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory = True,
            num_workers = 4
        )
    return dl
"""


def process_ast_tokens(ast_tokens_batch, tokenizer, max_ast_len, max_token_len=5):
    """
    批量处理一个 batch 内所有样本的 AST tokens
    """
    batch_tokenized_tensors = []
    for ast_tokens in ast_tokens_batch:  # 循环处理单个样本的 token 列表
        if len(ast_tokens) > 0:
            batch_encoding = tokenizer(
                ast_tokens,
                padding='max_length',
                truncation=True,
                max_length=max_token_len,
                return_tensors='pt',
                add_special_tokens = False
            )
            tokenized_tensor = batch_encoding['input_ids']
        else:
            tokenized_tensor = torch.zeros((0, max_token_len), dtype=torch.long)

        # Padding to max_ast_len
        pad_length = max_ast_len - tokenized_tensor.shape[0]
        if pad_length > 0:
            pad_tensor = torch.zeros((pad_length, max_token_len), dtype=torch.long)
            tokenized_tensor = torch.cat([tokenized_tensor, pad_tensor], dim=0)
        else:
            tokenized_tensor = tokenized_tensor[:max_ast_len]

        batch_tokenized_tensors.append(tokenized_tensor)

    return torch.stack(batch_tokenized_tensors, dim=0)  # [batch_size, max_ast_len, max_token_len]


def tokenization_batched(examples, args, tokenizer, context_tokenizer, max_seq_len, mode):
    if mode == 'diffusion' and args.dataset_name in {'xsum', 'qqp', 'wmt14-en-de', 'wmt14-de-en', 'cq_code',
                                                     'python_code', 'multiple_code'}:

        # 1. 批量 Tokenize context 和 text (性能提升的核心)
        if args.dataset_name in {'qqp', 'wmt14-en-de', 'wmt14-de-en', 'cq_code', 'python_code',
                                 'multiple_code'}:
            cond_inputs = context_tokenizer(examples['context'], padding="max_length", truncation=True,
                                            max_length=max_seq_len)
        elif args.dataset_name in {'xsum', }:
            cond_inputs = context_tokenizer(examples['context'], padding="max_length", truncation=True,
                                            max_length=max_seq_len * 4)
        else:
            raise NotImplementedError

        model_inputs = tokenizer(text_target=examples['text'], padding="max_length", truncation=True,
                                 max_length=max_seq_len)

        # 将 context 的 tokenization 结果添加到主输入中
        for k, v in cond_inputs.items():
            model_inputs[f'cond_{k}'] = v

        # 2. 循环处理 AST (只计算 ast_adj 和 ast_token_ids)
        all_ast_adj = []
        all_ast_tokens_raw = []  # 用于收集每个样本的原始 token 列表

        max_node = args.max_ast_len

        for ast_raw in examples['ast']:
            # ---------- 单个 AST 处理开始 ----------
            try:
                ast = json.loads(ast_raw) if isinstance(ast_raw, str) else []
            except (json.JSONDecodeError, TypeError):
                ast = []

            if not isinstance(ast, list) or not all(isinstance(node, dict) for node in ast):
                ast = []

            id2children = {node['id']: node['children'] for node in ast if 'children' in node}
            edges = [(parent, child) for parent, children in id2children.items() for child in children]

            num_nodes = len(ast)
            A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            for src, tgt in edges:
                if src < num_nodes and tgt < num_nodes:
                    A[src, tgt] = 1
                    A[tgt, src] = 1

            A = A + np.eye(num_nodes, dtype=np.float32)
            A_padded = pad_matrix(A, max_node)
            A_norm = normalize(A_padded)
            all_ast_adj.append(torch.FloatTensor(A_norm))

            ast_tokens = []
            for node in ast:
                node_type = node.get('type', '')
                node_val = node.get('value', '')
                if node_type:
                    ast_tokens.append(f"{node_type}_{node_val}" if node_val else node_type)
            all_ast_tokens_raw.append(ast_tokens)
            # ---------- 单个 AST 处理结束 ----------

        # 3. 批量处理 AST tokens
        ast_token_ids = process_ast_tokens(
            ast_tokens_batch=all_ast_tokens_raw,
            tokenizer=tokenizer,
            max_ast_len=args.max_ast_len
        )

        # 4. 将列表堆叠成 Batch 张量并添加到 model_inputs
        model_inputs['ast_adj'] = torch.stack(all_ast_adj, dim=0)
        model_inputs['ast_token_ids'] = ast_token_ids

        return model_inputs

    else:  # 非 diffusion 模式的简化逻辑
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_len)



def get_dataloader(args, split, dataset, tokenizer, max_seq_len, mode='diffusion', shuffle=True,
                   context_tokenizer=None):

    # --- DataLoader 创建部分保持不变 ---
    if 'mbart' in args.enc_dec_model:
        collate_fn = default_data_collator
    # elif 'bart' in args.enc_dec_model:
    #    collate_fn=DataCollatorForBartDenoisingLM(tokenizer, model_config.decoder_start_token_id)
    elif 'codet5' in args.enc_dec_model:
        collate_fn = DataCollatorForFlanLM(tokenizer)
    elif 't5' in args.enc_dec_model:
        collate_fn = DataCollatorForFlanLM(tokenizer)
    else:
        raise NotImplementedError

    map_fn_kwargs = {
        "args": args,
        "tokenizer": tokenizer,
        "context_tokenizer": context_tokenizer,
        "max_seq_len": max_seq_len,
        "mode": mode,
    }

    # 使用重构后的批处理函数
    if args.dataset_name in {'xsum', 'qqp'} or 'code' in args.dataset_name or 'cq_code' in args.dataset_name:
        dataset = dataset.map(
            tokenization_batched,  # Pass the global function
            remove_columns=['text', 'context', 'ast'],
            batched=True,  # Now batched=True will work
            fn_kwargs=map_fn_kwargs  # Pass extra arguments here
        )
    else:
        dataset = dataset.map(tokenization_batched, remove_columns='text', batched=True)

    if split == 'train':
        batch_size = args.train_batch_size
    else:
        batch_size = args.eval_batch_size

    dl = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=0
    )
    return dl

if __name__ == "__main__":

    dataset = get_dataset('roc')
    print(dataset['train'][0])