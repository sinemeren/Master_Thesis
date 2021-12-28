import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class Regression(BaseModel):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):

        out = self.l1(x)
        return out


class NN2Layer(BaseModel):
    def __init__(self, input_size, output_size, hidden_size_1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_size_1, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(l1)
        out = self.l2(l1_act)
        return out


class NN2Layer_Leaky(BaseModel):
    def __init__(self, input_size, output_size, hidden_size_1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.act1 = nn.LeakyReLU()
        self.l2 = nn.Linear(self.hidden_size_1, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(l1)
        out = self.l2(l1_act)
        return out


class NNMultiLayer(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.ReLU()
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(l1)
        l2 = self.l2(l1_act)
        l2_act = self.act2(l2)
        out = self.l3(l2_act)
        return out


class NNMultiLayerSELU(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.act1 = nn.SELU()
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.SELU()
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(l1)
        l2 = self.l2(l1_act)
        l2_act = self.act2(l2)
        out = self.l3(l2_act)
        return out


class NNMultiLayerSELU2(BaseModel):

    # no activation function in the last layer
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.act1 = nn.SELU()
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(l1)
        l2 = self.l2(l1_act)
        out = self.l3(l2)
        return out


class NNMultiLayerLeakyRELU(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.act1 = nn.LeakyReLU()
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.LeakyReLU()
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(l1)
        l2 = self.l2(l1_act)
        l2_act = self.act2(l2)
        out = self.l3(l2_act)
        return out


class NNMultiLayerLeakyRELU5(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, hidden_size_5):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.hidden_size_4 = hidden_size_4
        self.hidden_size_5 = hidden_size_5
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.act = nn.LeakyReLU()
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.l4 = nn.Linear(self.hidden_size_3, self.hidden_size_4)
        self.l5 = nn.Linear(self.hidden_size_4, self.hidden_size_5)
        self.l6 = nn.Linear(self.hidden_size_5, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act(l1)
        l2 = self.l2(l1_act)
        l2_act = self.act(l2)
        l3 = self.l3(l2_act)
        l3_act = self.act(l3)
        l4 = self.l4(l3_act)
        l4_act = self.act(l4)
        l5 = self.l5(l4_act)
        l5_act = self.act(l5)
        out = self.l6(l5_act)

        return out


class NNMultiLayerLeakyRELU_dropout(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, dropout):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.dropout = dropout

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.act = nn.LeakyReLU()
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.do = nn.Dropout(self.dropout)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act(self.bn1(l1))
        l2 = self.l2(l1_act)
        l2_act = self.act(l2)
        l2_do = self.do(l2_act)
        out = self.l3(l2_do)
        return out


class NNMultiLayer3LeakyRELU_dropout(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3, dropout):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size
        self.dropout = dropout

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.act = nn.LeakyReLU()
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.l4 = nn.Linear(self.hidden_size_3, self.output_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.do = nn.Dropout(self.dropout)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act(self.bn1(l1))
        l2 = self.l2(l1_act)
        l2_act = self.act(self.bn2(l2))
        l2_do = self.do(l2_act)
        l3 = self.l3(l2_do)
        l3_act = self.act(l3)
        out = self.l4(l3_act)
        return out


class NNMultiLayerLeakyRELU2(BaseModel):

    # no activation function in the last layer
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.act1 = nn.LeakyReLU()
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(l1)
        l2 = self.l2(l1_act)
        out = self.l3(l2)
        return out


class NNMultiLayerTanh(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.act1 = nn.Tanh()
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.Tanh()
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(l1)
        l2 = self.l2(l1_act)
        l2_act = self.act2(l2)
        out = self.l3(l2_act)
        return out


class NNMultiLayerBatchNorm(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.act2 = nn.ReLU()
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l2 = self.l2(l1_act)
        l2_act = self.act2(self.bn2(l2))
        out = self.l3(l2_act)
        return out


class NNMultiLayerBatchNorm_1(BaseModel):

    # no batch normalization in the last layer
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.ReLU()
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l2 = self.l2(l1_act)
        l2_act = self.act2(l2)
        out = self.l3(l2_act)
        return out


class NNMultiLayerBatchNorm_5(BaseModel):

    # no batch normalization in the last layer
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, hidden_size_5):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.hidden_size_4 = hidden_size_4
        self.hidden_size_5 = hidden_size_5
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.bn3 = nn.BatchNorm1d(self.hidden_size_3)
        self.bn4 = nn.BatchNorm1d(self.hidden_size_4)

        self.act = nn.LeakyReLU()

        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.l4 = nn.Linear(self.hidden_size_3, self.hidden_size_4)
        self.l5 = nn.Linear(self.hidden_size_4, self.hidden_size_5)
        self.l6 = nn.Linear(self.hidden_size_5, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act(self.bn1(l1))
        l2 = self.l2(l1_act)
        l2_act = self.act(self.bn2(l2))
        l3 = self.l3(l2_act)
        l3_act = self.act(self.bn3(l3))
        l4 = self.l4(l3_act)
        l4_act = self.act(self.bn4(l4))
        l5 = self.l5(l4_act)
        l5_act = self.act(l5)
        out = self.l6(l5_act)
        return out


class NNMultiLayerBatchNorm_5_dropOut(BaseModel):

    # no batch normalization in the last layer
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, hidden_size_5, dropoutRatio):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.hidden_size_4 = hidden_size_4
        self.hidden_size_5 = hidden_size_5
        self.output_size = output_size
        self.dropoutRatio = dropoutRatio

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.bn3 = nn.BatchNorm1d(self.hidden_size_3)
        self.bn4 = nn.BatchNorm1d(self.hidden_size_4)
        self.do = nn.Dropout(self.dropoutRatio)

        self.act = nn.LeakyReLU()

        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.l4 = nn.Linear(self.hidden_size_3, self.hidden_size_4)
        self.l5 = nn.Linear(self.hidden_size_4, self.hidden_size_5)
        self.l6 = nn.Linear(self.hidden_size_5, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act(self.bn1(l1))
        l2 = self.l2(l1_act)
        l2_act = self.act(self.bn2(l2))
        l3 = self.l3(l2_act)
        l3_act = self.act(self.bn3(l3))
        l4 = self.l4(l3_act)
        l4_act = self.act(self.bn4(l4))
        l4_do = self.do(l4_act)
        l5 = self.l5(l4_do)
        l5_act = self.act(l5)
        out = self.l6(l5_act)
        return out


class NNMultiLayerBatchNorm_3(BaseModel):

    # no batch normalization in the last layer
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)

        self.act = nn.LeakyReLU()

        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.l4 = nn.Linear(self.hidden_size_3, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act(self.bn1(l1))
        l2 = self.l2(l1_act)
        l2_act = self.act(self.bn2(l2))
        l3 = self.l3(l2_act)
        l3_act = self.act(l3)
        out = self.l4(l3_act)
        return out


class NNMultiLayerDropOutBatchNorm(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.ReLU()

        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.act2 = nn.ReLU()
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(self.bn2(l2))
        l2_do = self.do2(l2_act)
        out = self.l3(l2_do)

        return out


class NNMultiLayerDropOutBatchNorm_1(BaseModel):
    # no batch normalization in the last layer
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.ReLU()
        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.ReLU()
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(l2)
        l2_do = self.do2(l2_act)
        out = self.l3(l2_do)

        return out


class NNMultiLayerDropOutBatchNorm_LeakyRelu(BaseModel):
    # no batch normalization in the last layer
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.LeakyReLU()
        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.LeakyReLU()
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(l2)
        l2_do = self.do2(l2_act)
        out = self.l3(l2_do)

        return out


class NNMultiLayerDropOutBatchNorm_LeakyRelu_1(BaseModel):
    # no batch normalization in the last layer
    # dropout 0,6
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.LeakyReLU()
        self.do1 = nn.Dropout(0.6)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.LeakyReLU()
        self.do2 = nn.Dropout(0.6)
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(l2)
        l2_do = self.do2(l2_act)
        out = self.l3(l2_do)

        return out


class NNMultiLayerDropOutBatchNorm_LeakyRelu_DropOutRatio(BaseModel):
    # no batch normalization in the last layer
    # dropout 0,6
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, dropOutRatio):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.dropOutRatio = dropOutRatio

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.LeakyReLU()
        self.do1 = nn.Dropout(self.dropOutRatio)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.LeakyReLU()
        self.do2 = nn.Dropout(self.dropOutRatio)
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(l2)
        l2_do = self.do2(l2_act)
        out = self.l3(l2_do)

        return out


class NNMultiLayerDropOutBatchNorm_LeakyRelu_withDifferentDropOutRatio(BaseModel):
    # no batch normalization in the last layer
    # different dropout for the input and hidden layer
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.LeakyReLU()
        self.do1 = nn.Dropout(0.2)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.LeakyReLU()
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(l2)
        l2_do = self.do2(l2_act)
        out = self.l3(l2_do)

        return out


class NNMultiLayerDropOutBatchNorm3(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.ReLU()
        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.act2 = nn.ReLU()
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.bn3 = nn.BatchNorm1d(self.hidden_size_3)
        self.act3 = nn.ReLU()
        self.l4 = nn.Linear(self.hidden_size_3, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(self.bn2(l2))
        l2_do = self.do2(l2_act)
        l3 = self.l3(l2_do)
        l3_act = self.act3(self.bn3(l3))
        out = self.l4(l3_act)
        return out


class NNMultiLayerDropOutBatchNorm3SELU(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.SELU()
        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.act2 = nn.SELU()
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.bn3 = nn.BatchNorm1d(self.hidden_size_3)
        self.act3 = nn.SELU()
        self.l4 = nn.Linear(self.hidden_size_3, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(self.bn2(l2))
        l2_do = self.do2(l2_act)
        l3 = self.l3(l2_do)
        l3_act = self.act3(self.bn3(l3))
        out = self.l4(l3_act)
        return out


class NNMultiLayerDropOutBatchNorm3SELU_1(BaseModel):

    # without batch normalization in the last layer
    # activation function is SELU for all layers

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.SELU()
        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.act2 = nn.SELU()
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.act3 = nn.SELU()
        self.l4 = nn.Linear(self.hidden_size_3, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(self.bn2(l2))
        l2_do = self.do2(l2_act)
        l3 = self.l3(l2_do)
        l3_act = self.act3(l3)
        out = self.l4(l3_act)
        return out


class NNMultiLayerDropOutBatchNorm3LeakyReLu(BaseModel):

    # without batch normalization in the last layer
    # activation function is Leaky ReLu for all layers

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.LeakyReLU()
        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.act2 = nn.LeakyReLU()
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.act3 = nn.LeakyReLU()
        self.l4 = nn.Linear(self.hidden_size_3, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(self.bn2(l2))
        l2_do = self.do2(l2_act)
        l3 = self.l3(l2_do)
        l3_act = self.act3(l3)
        out = self.l4(l3_act)
        return out


class NNMultiLayerDropOutBatchNorm3LeakyReLu_dropoutRatio(BaseModel):

    # without batch normalization in the last layer
    # activation function is Leaky ReLu for all layers

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3, dropoutRatio):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size
        self.dropoutRatio = dropoutRatio

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.LeakyReLU()
        self.do1 = nn.Dropout(dropoutRatio)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.act2 = nn.LeakyReLU()
        self.do2 = nn.Dropout(dropoutRatio)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.act3 = nn.LeakyReLU()
        self.l4 = nn.Linear(self.hidden_size_3, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(self.bn2(l2))
        l2_do = self.do2(l2_act)
        l3 = self.l3(l2_do)
        l3_act = self.act3(l3)
        out = self.l4(l3_act)
        return out


class NNMultiLayerDropOutBatchNorm3_1(BaseModel):

    # without batch normalization in the last layer
    # activation function is ReLu for all layers

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.ReLU()
        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.act2 = nn.ReLU()
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.act3 = nn.ReLU()
        self.l4 = nn.Linear(self.hidden_size_3, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(self.bn2(l2))
        l2_do = self.do2(l2_act)
        l3 = self.l3(l2_do)
        l3_act = self.act3(l3)
        out = self.l4(l3_act)
        return out


class NNMultiLayerDropOutBatchNorm4(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.hidden_size_4 = hidden_size_4
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.ReLU()
        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.act2 = nn.ReLU()
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.bn3 = nn.BatchNorm1d(self.hidden_size_3)
        self.act3 = nn.ReLU()
        self.do3 = nn.Dropout(0.5)
        self.l4 = nn.Linear(self.hidden_size_3, self.hidden_size_4)
        self.bn4 = nn.BatchNorm1d(self.hidden_size_4)
        self.act4 = nn.ReLU()
        self.do4 = nn.Dropout(0.5)
        self.l5 = nn.Linear(self.hidden_size_4, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(self.bn2(l2))
        l2_do = self.do2(l2_act)
        l3 = self.l3(l2_do)
        l3_act = self.act3(self.bn3(l3))
        l3_do = self.do3(l3_act)
        l4 = self.l4(l3_do)
        l4_act = self.act4(self.bn4(l4))
        l4_do = self.do4(l4_act)
        out = self.l5(l4_do)

        return out


class NNMultiLayerDropOut(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.ReLU()
        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.act2 = nn.ReLU()
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(l1)
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(l2)
        l2_do = self.do2(l2_act)
        out = self.l3(l2_do)
        return out


class NNMultiLayerDropOutTrue(BaseModel):
    # dont have dropout layer at the output layer
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.act1 = nn.ReLU()
        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.act2 = nn.ReLU()
        self.l3 = nn.Linear(self.hidden_size_2, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(l1)
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(l2)
        out = self.l3(l2_act)
        return out


class NNMultiLayer3(BaseModel):
    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.ReLU()
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.act3 = nn.ReLU()
        self.l4 = nn.Linear(self.hidden_size_3, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(l1)
        l2 = self.l2(l1_act)
        l2_act = self.act2(l2)
        l3 = self.l3(l2_act)
        l3_act = self.act3(l3)
        out = self.l4(l3_act)
        return out


class NNMultiLayer3BatchNorm(BaseModel):
    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(self.hidden_size_3)
        self.l4 = nn.Linear(self.hidden_size_3, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l2 = self.l2(l1_act)
        l2_act = self.act2(self.bn2(l2))
        l3 = self.l3(l2_act)
        l3_act = self.act3(self.bn3(l3))
        out = self.l4(l3_act)
        return out


class ModelInPaper(BaseModel):
    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, hidden_size_5):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.hidden_size_4 = hidden_size_4
        self.hidden_size_5 = hidden_size_5
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(self.hidden_size_3)
        self.do3 = nn.Dropout(0.5)
        self.l4 = nn.Linear(self.hidden_size_3, self.hidden_size_4)
        self.act4 = nn.ReLU()
        self.l5 = nn.Linear(self.hidden_size_4, self.hidden_size_5)
        self.act5 = nn.ReLU()
        self.l6 = nn.Linear(self.hidden_size_5, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(self.bn2(l2))
        l2_do = self.do2(l2_act)
        l3 = self.l3(l2_do)
        l3_act = self.act3(self.bn3(l3))
        l3_do = self.do3(l3_act)
        l4 = self.l4(l3_do)
        l4_act = self.act4(l4)
        l5 = self.l5(l4_act)
        l5_act = self.act5(l5)
        out = self.l6(l5_act)

        return out


class ModelInPaperTrue(BaseModel):
    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, hidden_size_5):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.hidden_size_4 = hidden_size_4
        self.hidden_size_5 = hidden_size_5
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(self.hidden_size_3)
        self.do3 = nn.Dropout(0.5)
        self.l4 = nn.Linear(self.hidden_size_3, self.hidden_size_4)
        self.act4 = nn.LeakyReLU()
        self.l5 = nn.Linear(self.hidden_size_4, self.hidden_size_5)
        self.act5 = nn.ReLU()
        self.l6 = nn.Linear(self.hidden_size_5, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(self.bn2(l2))
        l2_do = self.do2(l2_act)
        l3 = self.l3(l2_do)
        l3_act = self.act3(self.bn3(l3))
        l3_do = self.do3(l3_act)
        l4 = self.l4(l3_do)
        l4_act = self.act4(l4)
        l5 = self.l5(l4_act)
        l5_act = self.act5(l5)
        out = self.l6(l5_act)

        return out


class ModelInPaperTrueSELU(BaseModel):
    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, hidden_size_5):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.hidden_size_4 = hidden_size_4
        self.hidden_size_5 = hidden_size_5
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.act1 = nn.SELU()
        self.bn1 = nn.BatchNorm1d(self.hidden_size_1)
        self.do1 = nn.Dropout(0.5)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.act2 = nn.SELU()
        self.bn2 = nn.BatchNorm1d(self.hidden_size_2)
        self.do2 = nn.Dropout(0.5)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.act3 = nn.SELU()
        self.bn3 = nn.BatchNorm1d(self.hidden_size_3)
        self.do3 = nn.Dropout(0.5)
        self.l4 = nn.Linear(self.hidden_size_3, self.hidden_size_4)
        self.act4 = nn.SELU()
        self.l5 = nn.Linear(self.hidden_size_4, self.hidden_size_5)
        self.act5 = nn.ReLU()
        self.l6 = nn.Linear(self.hidden_size_5, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act1(self.bn1(l1))
        l1_do = self.do1(l1_act)
        l2 = self.l2(l1_do)
        l2_act = self.act2(self.bn2(l2))
        l2_do = self.do2(l2_act)
        l3 = self.l3(l2_do)
        l3_act = self.act3(self.bn3(l3))
        l3_do = self.do3(l3_act)
        l4 = self.l4(l3_do)
        l4_act = self.act4(l4)
        l5 = self.l5(l4_act)
        l5_act = self.act5(l5)
        out = self.l6(l5_act)

        return out


class NNMultiLayerLeakyRELU3(BaseModel):

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, hidden_size_3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        # self.l1.apply(NNMultiLayer.init_weights)
        self.act = nn.LeakyReLU()
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.l4 = nn.Linear(self.hidden_size_3, self.output_size)

    def forward(self, x):

        l1 = self.l1(x)
        l1_act = self.act(l1)
        l2 = self.l2(l1_act)
        l2_act = self.act(l2)
        l3 = self.l3(l2_act)
        l3_act = self.act(l3)
        out = self.l4(l3_act)

        return out
