import torch
from secmlt.models.data_processing.data_processing import DataProcessing
from typing import Union, List

class BaseDocVQAProcessor(DataProcessing):
    """Base class for DocVQA processors, sharing common text tokenization methods."""

    def __init__(self):
        super().__init__()
        self.processor = None 
        self.target_suffix = "" 
        self.add_special_tokens = True 

    def get_input_ids(self, text: Union[List, str]):
        if text is None:
            return text
        if isinstance(text, str):
            return self.processor.tokenizer(text=text, return_tensors="pt").input_ids
            
        if isinstance(text, list) and len(text) > 0:
            tokenized_list = []

            for item in text:
                if self.target_suffix:
                    item += self.target_suffix
                
                tokenized = self.processor.tokenizer(
                    text=item, 
                    add_special_tokens=self.add_special_tokens, 
                    return_tensors="pt"
                ).input_ids
                
                tokenized_list.append(tokenized)
                tokenized_list.append(torch.tensor([-1], dtype=torch.long).unsqueeze(0))
            
            tokenized_list.pop() # remove the last separator
            res = torch.cat(tokenized_list, dim=1)
            return res
        else:
            raise ValueError("You have to put at least one element") 


    def reconstruct_targets(self, targets, separator=-1):
        res = []
        indices = (targets.flatten() == separator).to(dtype=torch.uint8).nonzero()
        
        if indices.numel() == 0: # edge case, just one target
            return targets

        start_index = 0
        for index in indices.squeeze(0):
            res.append(targets[:, start_index:index].squeeze(0))
            start_index = index + 1
        
        if indices.numel() != 0:
             res.append(targets[:, index+1:].squeeze(0))

        return res