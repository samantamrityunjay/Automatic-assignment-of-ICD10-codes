import torch.nn as nn
import transformers


class BERTclassifier(nn.Module):
    def __init__(self, bert_freeze=False):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 10)
        self.act = nn.Sigmoid()

        if bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, ids, mask, token_type_ids):
        outputs = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.drop(outputs[0][:, 0, :])
        x = self.out(x)
        x = self.act(x)
        return x
