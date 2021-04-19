import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

a = torch.tensor([1,2,3,4])
a = a.view(2,2)
b = torch.tensor([5,6,7,8,9,10])
b = b.view(-1,2)
c = torch.tensor([7,8,9,10])
c = c.view(-1,2)
d = torch.tensor([9,10])
d = d.view(-1,2)
train_x = [a, b, c, d]
class MyData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    seq_len = [s.size(0) for s in data] # 获取数据真实的长度
    data = pad_sequence(data, batch_first=True)    
    data = pack_padded_sequence(data, seq_len, batch_first=True)
    return data

# train_x.to_tensor()
print(train_x)

data = MyData(train_x)
data_loader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=collate_fn)
# 采用默认的 collate_fn 会报错
#data_loader = DataLoader(data, batch_size=2, shuffle=True) 
batch_x = iter(data_loader).next()
print(batch_x)