import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_ID = -1
EOS_ID = -1


class Attention(nn.Module):
    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False):
        '''实现了标准的注意力机制，用于在解码过程中对编码器输出的不同部分进行加权，
        以便将重要的部分突出显示，帮助生成更准确的输出。'''
        super(Attention, self).__init__()

        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        # input_dim：输入数据的维度（例如，词嵌入的维度或LSTM输出的维度）。
        # source_dim：源数据（编码器输出）的维度。
        # output_dim：输出的维度。
        # bias：是否使用偏置项。
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        #作用：
        # 初始化注意力机制的网络层，包括两个线性变换层：
        # input_proj：将输入数据投影到源数据维度。
        # output_proj：将输入数据和源数据的拼接结果投影到输出维度。
        # 初始化一个 mask 变量，用于掩盖不需要关注的部分（如填充符）。
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim, bias=bias)
        self.mask = None

    def set_mask(self, mask):
        #参数：
        # mask：一个掩码，用于指示哪些位置不应该被注意力机制关注（例如填充符）。
        # 作用：
        # 设置掩码，防止注意力机制关注不必要的部分。
        self.mask = mask

    def forward(self, input, source_hids):
        '''
        通过投影将输入数据映射到源数据维度，计算注意力分数并应用掩码（如果有）。
        计算加权和（加权的 source_hids），即对编码器输出的加权平均。
        将加权和与输入拼接后，通过输出投影生成最终的输出。
        :param input:
        :param source_hids:
        :return:
        '''
        batch_size = input.size(0)
        source_len = source_hids.size(1)

        # (batch, tgt_len, input_dim) -> (batch, tgt_len, source_dim)
        x = self.input_proj(input)

        # (batch, tgt_len, source_dim) * (batch, src_len, source_dim) -> (batch, tgt_len, src_len)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, source_len), dim=1).view(batch_size, -1, source_len)

        # (batch, tgt_len, src_len) * (batch, src_len, source_dim) -> (batch, tgt_len, source_dim)
        mix = torch.bmm(attn, source_hids)

        # concat -> (batch, tgt_len, source_dim + input_dim)
        combined = torch.cat((mix, input), dim=2)
        # output -> (batch, tgt_len, output_dim)
        output = torch.tanh(self.output_proj(combined.view(-1, self.input_dim + self.source_dim))).view(batch_size, -1,
                                                                                                        self.output_dim)
        # output：加权后的输出。
        # attn：计算得到的注意力分数。
        return output, attn


class Decoder(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length, gpu):
        '''
        layers：LSTM 层数。
        vocab_size：词汇表大小（即输出类别数）。
        hidden_size：隐藏层大小（即每个 LSTM 层的输出维度）。
        dropout：dropout 概率，用于防止过拟合。
        length：解码器生成的序列的最大长度。
        gpu：是否使用 GPU（0 表示不使用，1 表示使用）。
        作用：
        初始化解码器的参数，并创建一个嵌入层、dropout 层等。
        :param layers:
        :param vocab_size:
        :param hidden_size:
        :param dropout:
        :param length:
        :param gpu:
        '''
        super(Decoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.length = length  # total length to decode
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.sos_id = vocab_size - 1
        self.eos_id = vocab_size - 1
        self.gpu = gpu

    def forward(self, x, encoder_hidden=None, encoder_outputs=None):
        '''
        参数：
        x：输入数据（例如，在训练时是目标序列，推理时是 SOS（开始符号））。
        encoder_hidden：编码器的隐藏状态。
        encoder_outputs：编码器的输出。
        作用：
        对输入数据进行验证和预处理。
        通过 LSTM 网络逐步生成解码器输出，并使用注意力机制对编码器的输出进行加权。
        如果是推理阶段，逐步生成每个词；如果是训练阶段，使用给定的目标序列进行解码
                :param x:
                :param encoder_hidden:
                :param encoder_outputs:
                :return:
        '''

        ret_dict = dict()
        ret_dict[Decoder.KEY_ATTN_SCORE] = list()
        if x is None:  # if not given x, then we are inferring!
            inference = True
        else:
            inference = False
        x, batch_size, length = self._validate_args(x, encoder_hidden,
                                                    encoder_outputs,
                                                    self.gpu)  # hidden is layer-wise out, out is final output
        assert length == self.length
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([length] * batch_size)

        def decode(step_, step_output_, step_attn_):
            decoder_outputs.append(step_output_)
            ret_dict[Decoder.KEY_ATTN_SCORE].append(step_attn_)
            # if step_ % 2 == 0:  # sample index, should be in [1, index-1]
            #     index = step_ // 2 % 10 // 2 + 3
            #     symbols_ = decoder_outputs[-1][:, 1:index].topk(1)[1] + 1
            # else:  # sample operation, should be in [7, 11]
            #     symbols_ = decoder_outputs[-1][:, 7:].topk(1)[1] + 7
            symbols_ = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols_)

            eos_batches = symbols_.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step_) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols_

        decoder_input = x[:, 0].unsqueeze(1)
        for di in range(length):
            if not inference:
                decoder_input = x[:, di].unsqueeze(1)
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                          encoder_outputs)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output, step_attn)
            decoder_input = symbols

        ret_dict[Decoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[Decoder.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([h for h in encoder_hidden])
        else:
            encoder_hidden = encoder_hidden
        return encoder_hidden

    def _validate_args(self, x, encoder_hidden, encoder_outputs, gpu=0):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if x is None and encoder_hidden is None:
            batch_size = 1
        else:
            if x is not None:
                batch_size = x.size(0)
            else:
                batch_size = encoder_hidden[0].size(1)

        # set default input and max decoding length
        if x is None:
            x = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1).cuda(gpu)
            max_length = self.length
        else:
            max_length = x.size(1)

        return x, batch_size, max_length

    def infer(self, x, encoder_hidden=None, encoder_outputs=None):
        decoder_outputs, decoder_hidden, _ = self.forward(x, encoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden

    def forward_step(self, x: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        pass


class RNNDecoder(Decoder):

    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length, gpu
                 ):
        super(RNNDecoder, self).__init__(
            layers,
            vocab_size,
            hidden_size,
            dropout,
            length, gpu)

        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.init_input = None
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)  # class num + 1 eos

    def forward_step(self, x, hidden, encoder_outputs):
        batch_size = x.size(0)
        output_size = x.size(1)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        output, attn = self.attention(output, encoder_outputs)  # attention from decoder_output and encoder_output
        predicted_softmax = F.log_softmax(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn


def construct_decoder() -> Decoder:
    name = "rnn"
    size = 100
    info(f'Construct Decoder with method {name}...')
    if name == 'rnn':
        return RNNDecoder(
            layers=1,
            vocab_size=size + 1,
            hidden_size=64,
            dropout=0.0,
            length=size,
            gpu=0
        )
    else:
        assert False
