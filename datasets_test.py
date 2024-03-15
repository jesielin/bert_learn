from datasets import load_dataset,list_datasets,load_from_disk
from sklearn.model_selection import train_test_split

# datasets_list = list_datasets()
# print(len(datasets_list))
# print(datasets_list[:100])
#加载数据
dataset = load_from_disk(dataset_path='D:\work\PycharmProjects\Huggingface_Toturials\data\ChnSentiCorp')
print(dataset)

train_dataset = dataset['train']

print(train_dataset)

print(train_dataset[0])

print(train_dataset['label'][:10])

#排序
sorted_train_dataset = train_dataset.sort('label')
print(sorted_train_dataset['label'][:10])
print(sorted_train_dataset['label'][-10:])

#打乱顺序
shuffle_train_dataset = train_dataset.shuffle(seed=42)
print(shuffle_train_dataset['label'][:10])

#选择
selected_train_dataset = train_dataset.select([0,10,20,30,40,50])
print(selected_train_dataset)

#过滤
start_filter_dataset = train_dataset.filter(lambda x:x['text'].startswith('选择'))
print(start_filter_dataset)
for i in start_filter_dataset:
    print(i)

#切分
train_text,test_text = train_test_split(train_dataset['text'],train_size=0.8,random_state=42)
train_dataset_new = train_dataset.filter(lambda x:x['text'] in train_text)
test_dataset_new = train_dataset.filter(lambda x:x['text'] in test_text)
print(train_dataset_new)
print(test_dataset_new)

#分桶
# shard_dataset = dataset.shard(num_shards=4,index=0)
# print(shard_dataset)

def fn(data):
    data['text'] = 'My sentence:'+data['text']
    return data

#map
train_dataset_map = train_dataset.map(fn)
print(train_dataset_map['text'][:5])

#保存和加载
dataset.save_to_disk('./')
current_dataset = dataset.load_from_disk('./')
print(current_dataset)