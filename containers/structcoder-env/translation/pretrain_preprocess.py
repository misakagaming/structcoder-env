#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import load_dataset
from tqdm import tqdm
from parser import remove_comments_and_docstrings
import pandas as pd
tqdm.pandas()
import numpy as np
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index, 
                   detokenize_code, tree_to_token_nodes)
from tree_sitter import Language, Parser
import pickle
import os
from transformers import RobertaTokenizer
import matplotlib.pyplot as plt


def read_pt_dataset(max_samples_per_split=None):
    dataset = load_dataset('code_search_net')
    rows = []
    
    for split in ['train', 'test', 'validation']:
        num_samples_in_split = len(dataset[split])
        indices = list(range(num_samples_in_split))
        if (max_samples_per_split is not None) and (num_samples_in_split>max_samples_per_split):
            indices = list(map(int, np.random.choice(indices, max_samples_per_split, replace=False)))
        pbar = tqdm(indices)
        pbar.set_description('Reading split='+split)
        
        for i in pbar:
            sample = dataset[split][i]
            rows.append([sample['func_code_string'], sample['language'], 
                         sample['func_documentation_string']])
            
    return pd.DataFrame(rows, columns=['code', 'lang', 'text'])


def add_php_ends(code):
    if not(code.startswith('<?php')):
        code="<?php "+code
    if not(code.endswith('?>')):
        code=code+"?>" 
    return code


def print_lang_dist(langs, total=None):
    if total is None:
        total = len(langs)
    vc = pd.value_counts(langs)
    display(pd.DataFrame({'lang':vc.index, 'count':vc.values, 'perc':vc.values/total*100}))

    
def get_tokenizer_chars():
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    tokenizer_chars = []
    for i in range(tokenizer.vocab_size):
        token = tokenizer.decode(i)
        if len(token)==1:
            tokenizer_chars.append(token)
    tokenizer_chars = [c for c in tokenizer_chars if c!='�']
    return tokenizer_chars
    

def preprocess(data):
    codes = []
    failed_count = 0
    failed_langs = []
    rows = []
    tokenizer_chars = get_tokenizer_chars()
    pbar = tqdm(data.itertuples())
    for row in pbar:
        code = row.code.strip().replace('▁', '_').replace('\r\n', '\n') # step 1
        code = ''.join(filter(lambda c:c in tokenizer_chars, code)) # step 2
        if row.lang=="php":
            code = add_php_ends(code) # step 3
        try:
            code = remove_comments_and_docstrings(code, row.lang) # step 4
        except:
            failed_count += 1
            failed_langs.append(row.lang)
            pbar.set_description('failed_count='+str(failed_count))
            continue
        rows.append([code, row.lang, row.text.strip()])
    if failed_count:
        print ('Distribution of languages among failed samples for remove_comments_and_docstrings()')
        print_lang_dist(failed_langs)
    data = pd.DataFrame(rows, columns=['code', 'lang', 'text'])
    print ('Distribution of languages after removing samples failing remove_comments_and_docstrings()')
    print_lang_dist(data.lang)
    return data


def extract_structure(code, parser):
    # ast
    tree = parser[0].parse(bytes(code,'utf8'))    
    root_node = tree.root_node  
    ast_token_nodes = tree_to_token_nodes(root_node) # leaves
    
    # dfg
    tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
    code=code.split('\n')
    code_tokens=[index_to_code_token(x,code) for x in tokens_index] 
    index_to_code={index:(idx,code_) for idx,(index,code_) in enumerate(zip(tokens_index,code_tokens))}
    try:
        DFG,_ = parser[1](root_node,index_to_code,{}) 
    except:
        DFG = []
    for d in DFG:
        assert (d[2]=='comesFrom' or d[2]=='computedFrom')
    DFG = [(d[1], d[4]) for d in DFG if (len(d[4])>0)] # left comes from right
    return code_tokens, ast_token_nodes, DFG


def format_node_ranges(code, nodes):
    line_lens = [len(line)+1 for line in code.split('\n')]
    line_starts = [0] + list(np.cumsum(line_lens))
    return [(line_starts[node.start_point[0]]+node.start_point[1],
             line_starts[node.end_point[0]]+node.end_point[1]) for node in nodes]

    
def length_stats(s, title=None):
    try:
        if type(s.iloc[0])==str: # a list encoded as str
            lens = s.apply(lambda x:x.count(',')+1)
        else: # a list
            lens = s.apply(len)
    except:
        lens = s # s contains lengths
    y = np.arange(100)
    x = lens.quantile(y/100)
    plt.figure()
    plt.plot(x,y)
    plt.title(title)
    plt.show()
    
    
def add_structure(data):
    dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'php':DFG_php,
    'javascript':DFG_javascript,
    'go':DFG_go,
    'ruby':DFG_ruby
    }

    parsers={}        
    for lang in dfg_function:
        LANGUAGE = Language('parser/my-languages2.so', lang)
        parser = Parser()
        parser.set_language(LANGUAGE) 
        parser = [parser,dfg_function[lang]]    
        parsers[lang]= parser
        
    ast_leaf_tokens, ast_leaves, ast_leaf_ranges, dfg_edges = [], [], [], []
    for row in tqdm(data.itertuples()):
        curr_code_tokens, curr_ast_leaves, curr_dfg_edges = extract_structure(row.code, parsers[row.lang])
        ast_leaf_tokens.append(curr_code_tokens)
        ast_leaves.append(curr_ast_leaves)
        ast_leaf_ranges.append(format_node_ranges(row.code, curr_ast_leaves))
        dfg_edges.append(curr_dfg_edges)
        
    data['ast_leaves'] = ast_leaves # list of leaf nodes
    data['dfg_edges'] = dfg_edges # list of "left leaf node index comes from right leaf nodes indices"
    data['ast_leaf_tokens'] = ast_leaf_tokens # list of code substrings corresponding to each leaf
    data['ast_leaf_ranges'] = ast_leaf_ranges # list of (start,end) in code for each leaf node
    
    print ('Distribution of languages among codes with failed/empty DFG')
    print_lang_dist(data.loc[data['dfg_edges'].apply(len)==0].lang, total=len(data))
    
    
def tokenize_codes_texts(texts, batch_size=1024):
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    N = len(texts)
    tokenized_texts = []
    for start in tqdm(range(0, len(texts),batch_size)):
        tokenized_texts += tokenizer(texts[start:start+batch_size]).input_ids
    return tokenized_texts

    
def get_code_tokens_ranges(data):
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    match  = {lang:[0,0] for lang in data['lang'].unique()}
    pbar = tqdm(data.itertuples())
    ranges = []
    
    for row in pbar:
        code_tokens = [tokenizer.decode(ct) for ct in row.code_tokens][1:-1] # 1:-1 to remove <s> and </s>
        code2 = ''.join(code_tokens) # misses some spaces that are in row.code
        code = row.code
        
        # map each position in code2 to a position in code
        code2_to_code = []
        j=0
        for i in range(len(code2)):
            if code2[i]==code[j]:
                code2_to_code.append(j)
                j += 1
            elif code2[i]==code[j+1]: # if code2 missed a space
                code2_to_code.append(j+1)
                j += 2
            else:
                raise Exception('Character "'+code2[i]+'" from tokenized code not found in code.')
            
        # map each code token to a range in code
        code2_idx = 0
        curr_ranges = []
        for ct in code_tokens:
            s,e = code2_idx, code2_idx+len(ct)
            code2_idx = e
            curr_ranges.append((min(code2_to_code[s:e]),1+max(code2_to_code[s:e])))
        ranges.append([None]+curr_ranges+[None]) # first and last for <s> and </s>
        
    data['code_tokens_ranges'] = ranges
    
    
def overlap(s1,e1,s2,e2):
    return s1<=s2<e1 or s2<=s1<e2
    
def get_leaf_code_token_indices(data):
    ast_leaf_token_idxs = []
    for row in tqdm(data.itertuples()):
        j = 1
        ast_leaf_token_idxs.append([])
        code_tokens_last_idx = len(row.code_tokens)-1
        for s,e in row.ast_leaf_ranges:
            if s==e: # there are leaves with start_point=end_point
                ast_leaf_token_idxs[-1].append([])
                continue
            while not(overlap(s,e,row.code_tokens_ranges[j][0],row.code_tokens_ranges[j][1])):
                j += 1
            jj = j
            curr_leaf_token_idxs = []
            while overlap(s,e,row.code_tokens_ranges[jj][0],row.code_tokens_ranges[jj][1]):
                curr_leaf_token_idxs.append(jj)
                jj += 1
                if jj==code_tokens_last_idx:
                    break
            ast_leaf_token_idxs[-1].append(curr_leaf_token_idxs)
    data['ast_leaf_code_token_idxs'] = ast_leaf_token_idxs
    

def get_lr_path(leaf):
    path = [leaf]
    while path[-1].parent is not None:
        path.append(path[-1].parent)
    return path


def get_ll_sim(p1, p2): 
    common = 1
    for i in range(2, min(len(p1), len(p2))+1):
        if p1[-i]==p2[-i]:
            common += 1
        else:
            break
    return common*common / (len(p1)*len(p2))   


def process_dfg_edges(data):
    dfg_node_code_token_idxs = []
    dfg_edges = []
    for row in tqdm(data.itertuples()):
        if len(row.dfg_edges)>0:
            nodes = sorted(list(set(np.concatenate([[left]+right for left,right in row.dfg_edges]))))
        else:
            nodes = []
        node_to_idx = {k:i for i,k in enumerate(nodes)}
        dfg_node_code_token_idxs.append( [row.ast_leaf_code_token_idxs[i] for i in nodes] )
        dfg_edges.append( [(node_to_idx[left], [node_to_idx[r] for r in right]) for left,right in row.dfg_edges] )
    data['dfg_edges'] = dfg_edges
    data['dfg_node_code_token_idxs'] = dfg_node_code_token_idxs
    
def get_ast_lr_paths_and_ll_sim(data):
    sims = []
    lr_paths = []
    all_node_types = set()
    for i,row in tqdm(enumerate(data.itertuples())):
        L = min(len(row.ast_leaves), 512)
        curr_paths = [get_lr_path(leaf) for leaf in row.ast_leaves]
        curr_sims = np.ones((L,L))
        for i in range(L-1):
            for j in range(i+1,L):
                curr_sims[i,j] = curr_sims[j,i] = get_ll_sim(curr_paths[i], curr_paths[j])
        sims.append(';'.join([','.join(list(map(str,row))) for row in curr_sims]))
        lr_paths.append([[node.type for node in path] for path in curr_paths])
        all_node_types.update(set(np.concatenate(lr_paths[-1])))
    data.drop(columns=['ast_leaves'], inplace=True)
    data['ll_sims'] = sims
    data['lr_paths_types'] = lr_paths
    return all_node_types

def parse_list_of_lists(s, type_=int):
    list_of_lists = s[1:-2].split('], ')
    if type_==str:
        list_of_lists = [[t[1:-1].replace('\\n','\n').replace('\\\\','\\') for  t in x[1:].split(', ')] \
                         for x in list_of_lists]
    elif type_==int:
        list_of_lists = [[int(t) for  t in x[1:].split(', ')] for x in list_of_lists]
    else:
        raise Exception('Unknown value for type_')
    return list_of_lists


# In[ ]:


num_samples_per_split, num_rows_per_file = None, 10000
# num_samples_per_split, num_rows_per_file = 100, 200 # for debugging

np.random.seed(10)
data = read_pt_dataset(num_samples_per_split) # columns: code, text, lang
data = preprocess(data)
add_structure(data) # columns: ast_leaves, dfg_edges, ast_leaf_tokens, ast_leaf_ranges
data['code_tokens'] = tokenize_codes_texts(list(data['code']))
data['text_tokens'] = tokenize_codes_texts(list(data['text']))
length_stats(data['code_tokens'], 'Distribution of #code_tokens')
length_stats(data['text_tokens'], 'Distribution of #text_tokens')
get_code_tokens_ranges(data) # columns: code_token_ranges -> list of (start,end) one for each code_token
data.drop(columns=['code', 'text'], inplace=True)
get_leaf_code_token_indices(data)
data.drop(columns=['ast_leaf_tokens', 'ast_leaf_ranges', 'code_tokens_ranges'], inplace=True)
for col in ['code_tokens', 'text_tokens']:
    data[col] = data[col].progress_apply(lambda l:','.join(list(map(str,l))))
data = data.sample(frac=1).reset_index(drop=True)
# columns -> ['lang', 'ast_leaves', 'dfg_edges', 'code_tokens', 'text_tokens', 'ast_leaf_code_token_idxs']

# do memory intensive part in chunks
save_dir = 'data/pretrain/'
os.makedirs(save_dir, exist_ok=True)
all_node_types = set()
for start in range(0,len(data),num_rows_per_file):
    print ('Working on from_'+str(start)+'.parquet')
    sub_data = data.iloc[start:start+num_rows_per_file].copy() # copy so that edits are not on data
    sub_node_types = get_ast_lr_paths_and_ll_sim(sub_data)
    all_node_types.update(sub_node_types)
    process_dfg_edges(sub_data)
    sub_data = sub_data[['code_tokens', 'text_tokens', 'lang', 
                         'ast_leaf_code_token_idxs', 'll_sims', 'lr_paths_types', 
                         'dfg_node_code_token_idxs', 'dfg_edges']]
    for col in ['ast_leaf_code_token_idxs', 'lr_paths_types', 'dfg_node_code_token_idxs', 'dfg_edges']:
        sub_data[col] = sub_data[col].apply(str)
    sub_data.to_parquet(save_dir+'from_'+str(start)+'.parquet', engine='fastparquet', row_group_offsets=100)
del data
    
# convert node types to indices
all_node_types = sorted(list(all_node_types))
node_type_to_ind = {t:i for i,t in enumerate(all_node_types)}
pickle.dump(all_node_types, open(save_dir+'all_node_types.pkl', 'wb'))

for filename in tqdm(os.listdir(save_dir)):
    if filename.startswith('from_'):
        sub_data = pd.read_parquet(save_dir+filename, engine='fastparquet')
        sub_data['lr_paths_types'] = sub_data['lr_paths_types'].apply(
                        lambda s:str([[node_type_to_ind[t] for t in path] 
                                      for path in parse_list_of_lists(s, type_=str)]))
        sub_data.to_parquet(save_dir+filename, engine='fastparquet', row_group_offsets=100)


# In[ ]:


# Reduce memory taken by ll_sims column by storing only upper triangles w/o diagnoals.
def upper_triangle(s):
    rows = s.split(';')[:-1] 
    s = ''
    for i,row in enumerate(rows):
        s += ','.join(row.split(',')[i+1:]) + ';'
    return s[:-1]
pbar = tqdm(os.listdir(save_dir))
for filename in pbar:
    pbar.set_description(filename)
    if filename.startswith('from_'):
        sub_data = pd.read_parquet(save_dir+filename, engine='fastparquet')
        sub_data['ll_sims'] = sub_data['ll_sims'].apply(upper_triangle)
        sub_data.to_parquet(save_dir+filename, engine='fastparquet', row_group_offsets=100)


# In[ ]:


def some_more_stats(data):
    data['lr_paths_types'] = data['lr_paths_types'].progress_apply(lambda s:parse_list_of_lists(s, type_=int))
    node_types= pickle.load(open('data/pretrain/all_node_types.pkl','rb'))
    if 'ERROR' in node_types:
        error_node_idx = node_types.index('ERROR')
        num_error_nodes = data['lr_paths_types'].apply(lambda paths:np.mean([(np.array(path)==error_node_idx).max() 
                                                                      for path in paths]))
        print ('Distrubution of fraction of leaf-root paths with ERROR node in one code')
        length_stats(num_error_nodes)
    print ('Distrubution of AST depth')
    length_stats(data['lr_paths_types'].apply(lambda paths:max([len(p) for p in paths])))        
        
    print ('Distrubution of # ast leaves per code')
    length_stats(data['ast_leaf_code_token_idxs'].apply(lambda s:1+s.count('],')))
    print ('Distrubution of # dfg nodes per code')
    length_stats(data['dfg_node_code_token_idxs'].apply(lambda s:1+s.count('],')))
    print ('Distrubution of # dfg edges per code')
    def num_dfg_edges(s):
        if s=='[]':
            return 0
        return sum([t.split(', ',maxsplit=1)[1].count(',')+1 for t in s[1:-2].split('),')])
    length_stats(data['dfg_edges'].apply(num_dfg_edges))
    
data = []
save_dir = 'data/pretrain/'
for filename in tqdm(os.listdir(save_dir)):
    if filename.startswith('from_'):
        sub_data = pd.read_parquet(save_dir+filename, engine='fastparquet')
        data.append(sub_data)
        
data = pd.concat(data)
    
some_more_stats(data)


# In[ ]:





# In[ ]:


from sys import getsizeof
getsizeof(data)/1e6


# In[ ]:


for col in data.columns:
    print (col, getsizeof(data[col])/1e6)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




