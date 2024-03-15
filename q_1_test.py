import torch
from datasets import load_dataset,load_from_disk
from transformers import BertTokenizer,BertModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
class Dataset(torch.utils.data.Dataset):   
    def __init__(self, split):
        super().__init__()
        self.dataset = load_from_disk(dataset_path='D:\work\PycharmProjects\Huggingface_Toturials\data\ChnSentiCorp')[split]





    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        text = self.dataset[index]['text']
        label = self.dataset[index]['label']
        return text,label

def collate_fn(data):
    # 数据处理
    # 返回处理后的数据
    texts = [i[0] for i in data]
    labels = [i[1] for i in data]
    tokenizer = BertTokenizer.from_pretrained(r'D:\work\bert')
    result = tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts,
                                truncation=True,
                                padding='max_length',
                                max_length=300,
                                return_tensors='pt',
                                return_length=True)
    input_ids = result['input_ids']
    attention_mask = result['attention_mask']
    token_type_ids = result['token_type_ids']
    labels = torch.LongTensor(labels)
    return input_ids,attention_mask,token_type_ids,labels

class MyTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classify = torch.nn.Linear(768,2)


    def forward(self,x,labels=None):


        out = x.last_hidden_state[:,0,:]
        pred = self.classify(out)
        if labels is not None:

            loss = torch.nn.CrossEntropyLoss()(pred,labels)
            return loss
        else:
            return pred




if __name__ == '__main__':
    dataset = Dataset('train')
    print(dataset[:2])
    dl = DataLoader(dataset=dataset,collate_fn=collate_fn,batch_size=16,shuffle=False,drop_last=True)

    mymodel = MyTorchModel()
    # mymodel.eval()

    pretrain_model = BertModel.from_pretrained(r'D:\work\bert')
    # 不计算梯度
    for i in pretrain_model.parameters():
        i.requires_grad = False


    optizer = torch.optim.Adam(mymodel.parameters(),lr=1e-3)
    if torch.cuda.is_available():
        mymodel = mymodel.cuda()
        pretrain_model = pretrain_model.cuda()
    mymodel.train()

    for epoch in range(10):
        loss_watch = []
        for index,batch_data in enumerate(dl):
            # print('index:',index)
            input_ids,attention_mask,token_type_ids,labels = batch_data
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                labels = labels.cuda()
            # print('input_ids:',input_ids.shape)
            # print('attention_mask:',attention_mask.shape)
            # print('token_type_ids:',token_type_ids.shape)
            # print('labels:',labels.shape)
            with torch.no_grad():
                out = pretrain_model(input_ids=input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids)


            loss = mymodel(out,labels=labels)
            # print('loss:', loss)
            loss_watch.append(loss.item())
            optizer.zero_grad()
            loss.backward()
            optizer.step()

            if index %10==0:
                out = mymodel(out)
                print('out:',out.shape)
                out = out.argmax(dim=-1)
                print((out==labels).sum().item())
                accuracy = (out==labels).sum().item()/len(labels)


                print('loss mean:',np.mean(loss_watch),'正确率：',accuracy,'index:',index)


