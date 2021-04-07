import numpy as np

from torch.utils.data import Dataset

class AKTDataset(Dataset):
    def __init__(self, group, n_question, max_seq=200, min_samples=1):
        super(AKTDataset, self).__init__()
        
        self.max_seq = max_seq
        self.n_question = n_question
        self.samples = {}
        self.user_ids = []
        
        for user_id in group.index:
            q_, res_ = group[user_id]
            if len(q_) < min_samples:
                continue
            
            seq_length = len(q_)
            if seq_length > self.max_seq:
                initial = seq_length % self.max_seq
                
                if initial >= min_samples:
                    self.user_ids.append(f"{user_id}_0")
                    q = q_[:initial]
                    res = res_[:initial]
                    self.samples[f"{user_id}_0"] = (q, res)
                    
                for seq in range(seq_length // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq+1}")
                    start = initial + seq * self.max_seq
                    end = start + self.max_seq
                    q = q_[start:end]
                    res = res_[start:end]
                    self.samples[f"{user_id}_{seq+1}"] = (q, res)
                    
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (q_, res_)
                
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, res_ = self.samples[user_id]
        seq_length = len(q_)

        q_ = q_ + 1
        
        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        res = np.zeros(self.max_seq, dtype=int)
        
        if seq_length >= self.max_seq:
            q[:] = q_[-self.max_seq:]
            res[:] = res_[-self.max_seq:]
        else:
            q[-seq_length:] = q_
            res[-seq_length:] = res_
        
        q = q[1:]
        res = res[1:]
        qa = res.astype(int) * self.n_question + q
        
        return q, qa, res