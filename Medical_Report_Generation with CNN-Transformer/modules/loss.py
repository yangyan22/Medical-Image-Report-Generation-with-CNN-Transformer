import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]  # target and input
        mask = mask[:, :input.size(1)]
        # print(input.shape)  # torch.Size([4, 59, 738])
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask  # the probability of generating target word out[i][j][k] = input[i][j][index[i][j][k]] dim=2
        # print(target.long().unsqueeze(2).shape)  # torch.Size([4, 59, 1])
        # print(torch.sum(mask))  # the number of 1
        # print(output.shape)  # torch.Size([4, 59])
        output = torch.sum(output) / torch.sum(mask)  # tensor(6.8164, device='cuda:0', grad_fn=<DivBackward0>)  
        return output


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])  # .mean() 
    return loss
