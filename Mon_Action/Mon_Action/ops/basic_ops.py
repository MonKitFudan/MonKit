import torch
from torch.nn.parameter import Parameter

class Identity(torch.nn.Module):
    def forward(self, input):
        return input

class SegmentConsensus(torch.nn.Module):

    def __init__(self, consensus_type, dim=1,conv1=None):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        # self.conv1 = conv1

        # self.conv1.weight = Parameter(torch.tensor([[[1.0]]],requires_grad=True).cuda())
        # self.conv1.weight = Parameter(torch.tensor([[[0.1641],
        #                                              [0.2315],
        #                                              [0.2584],
        #                                              [0.2826],
        #                                              [0.2956],
        #                                              [0.2978],
        #                                              [0.2746],
        #                                              [0.2270]]], requires_grad=True).cuda())


    def forward(self, input_tensor):
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'max':
            output = torch.max(input_tensor,dim=self.dim, keepdim=True)[0]
        elif self.consensus_type == 'wavg':
            # print(list(self.conv1.named_parameters()))
            # output = input_tensor
            output = self.conv1(input_tensor)

        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim
        # self.conv1 = torch.nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1, bias=False).cuda()
        # torch.nn.init.xavier_uniform(self.conv1.weight)
        # torch.nn.init.constant_(self.conv1.weight, 0.125)


    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)#,self.conv1
