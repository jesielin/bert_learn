from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(r'D:\work\bert-base-chinese', cache_dir=None, force_download=False)

sentences = [
    '保姆级教程带你彻底啃透AI顶会论文！',
    'BERT模型实战教程：基于BERT模型的文本分类、情感分析及中文命名实体识别实战教程，迪哥带你玩转NLP核心框架—BERT模型！（人工智能/深度学习）',
    'B站强推！2023公认最通俗易懂的HuggingFace教程，(安装+讲解+实战)一步到位！NLP预训练模型_Transformers_datasets类',
    '手把手教你用Pytorch代码实现Transformer模型',
    '强推！这可能是B站最全的HuggingFace简明教程了，精细讲解入门到实战 学完即可就业！人工智能_NLP预训练模型_Transformer模型_深度学习',
    '环境安装，BERT、GPT、T5 性能测试，和横向对比100亿模型计划'
]

#编码两个句子
out = tokenizer.encode(text=sentences[0],text_pair=sentences[3],
                       #当句子大于max length时截断
                        truncation=True,
                       #一律补满为max length
                       padding='max_length',

                       add_special_tokens=True,
                       max_length=50,
                       return_tensors=None
                       )

print(out)

print(tokenizer.decode(out))

#增强的编码函数
out = tokenizer.encode_plus(text=sentences[0],text_pair=sentences[3],
                            #当句子大于max length时截断
                            truncation=True,

                            #一律补满为max length
                            padding='max_length',
                            max_length=60,


                            add_special_tokens=True,

                            #可取tf(tensorflow),pt(pytorch),np(numpy) ,默认返回list
                            return_tensors=None,

                            return_token_type_ids=True,

                            return_attention_mask=True,

                            return_special_tokens_mask=True,

                            #返回offset_mapping,标识每个词的起始位置，这个参数只能在BertTokenizerFast中使用
                            #return_offsets_mapping=True,



                            #返回length标识长度
                            return_length=True)
#input_ids 就是编码后的词
#token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
#special_tokens_mask 特殊符号的位置是1,其他位置是0
#attention_mask pad的位置是0,其他位置是1
#length 返回句子长度

print(out)
print(type(out))
print(tokenizer.decode(out['input_ids']))

#批量编码句子
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[sentences[0], sentences[3]],
    add_special_tokens=True,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=60,

    #可取值tf,pt,np,默认为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

for k, v in out.items():
    print(k, ':', v)

print(tokenizer.decode(out['input_ids'][0]))
print(tokenizer.decode(out['input_ids'][1]))

#批量编码成对的句子
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[(sentences[0], sentences[1]), (sentences[2], sentences[3])],
    add_special_tokens=True,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=60,

    #可取值tf,pt,np,默认为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

#input_ids 就是编码后的词
#token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
#special_tokens_mask 特殊符号的位置是1,其他位置是0
#attention_mask pad的位置是0,其他位置是1
#length 返回句子长度
for k, v in out.items():
    print(k, ':', v)

for i,input_id in enumerate(out['input_ids']):
    print(f'{i}:{tokenizer.decode(input_id)}')

#获取bert词表

bert_vocab = tokenizer.get_vocab()
print(bert_vocab)
print(len(bert_vocab))

print(tokenizer.special_tokens_map)
tokenizer.add_special_tokens({'eos_token':'[EOS]'})
tokenizer.add_tokens(['你好','月光'])
bert_vocab = tokenizer.get_vocab()
print(bert_vocab)
print(len(bert_vocab))
print(tokenizer.encode('你好月光啊啊啊[EOS]'))
print(tokenizer.decode(tokenizer.encode('你好月光啊啊啊[EOS]')))


