import torch
import torch.nn

class rnn_model(nn.Model):
  def __init__(self, input_size, hidden_size, output_size, n_laryers = 4, dropout_p = 0.15):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p

    self.embedding = nn.Embedding(self.input_size, self.hidden_size)
    self.gru = nn.gru(input_size = self.hidden_size, 
                      hidden_size = self.hidden_size,
                      dropput = self.dropout_p,
                      n_layers = self.n_layers,
                      bidirectional = True
                      )
    self.relu = nn.Relu()
    self.log_softmax = nn.Log_Softmax()
    self.linear1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.linear2 = nn.Linear(self.hidden_size, self.output_size)

  def forward(input, hidden):
    embd_input = self.embedding(input)
    rnn_output, rnn_hidden = self.gru(embd_input)
    rnn_output = nn.Relu(rnn_output)
    rnn_output = self.linear1(rnn_output)
    rnn_output = self.linear2(rnn_output)
    log_soft_output = self.log_softmax(rnn_output)

    return log_soft_output