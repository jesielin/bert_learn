from transformers import pipeline
import torch
from transformers import BertTokenizer
# classifier = pipeline("sentiment-analysis")
#
# result = classifier("I love this car")
#
# print(result)
#
# result = classifier('I hate this car')
# print(result)

input_ids = torch.LongTensor([[101, 6848, 2885, 4403, 3736, 5709, 1736, 4638, 1333, 1728, 2218, 3221, 3175, 912, 102], [101, 5011, 6381, 3315, 4638, 7241, 4669, 4802, 2141, 4272, 511, 102, 0, 0, 0]])

labels = input_ids[:,5].clone()
token = BertTokenizer.from_pretrained(r'D:/work/bert')
input_ids[:,5] = token.get_vocab()[token.mask_token]
print(input_ids)
print(labels)