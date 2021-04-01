import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import easydict

import korean_semantic_classification_trainer as T
import korean_semactic_classification_Dataloader as D
import korean_semantic_classification_RNN_model as R

def define_argparser(): # colab에서는 easydict를 사용했습니다.
  args = easydict.EasyDict({
      'model_fn' : "",
      'train_fn' : '',
      'gpu' : -1,
      'verbose' : 2,
      'min_vocab_freq' : 5,
      'max_vocab_size' : 999999,
      'batch_size' : 256,
      'n_epochs' : 10,
      'word_vec_size' : 256,
      'dropout' : .3,
      'max_length' : 256,
      'hidden_size' : 512,
      'n_layers' : 4
  })
  config = args
  return config

def main(config):
  loaders = D.Dataloader(
      train_fn = config.train_fn,
      batch_size = config.batch_size,
      device = config.gpu,
      min_freq = config.min_vocab_freq,
      max_vocab = config.max_vocab_size,
  )

  print(
      '|train| = ', len(loaders.train_loader.dataset),
      '|valid| = ', len(loaders.valid_loaders.dataset),
  )

  vocab_size = len(loaders.text.vocab)
  n_classes = len(loaders.label.vocab)
  print('|vocab| =', vocab_size, '|valid| =', n_classes)

  model = R.rnn_model(
      input_size = vocab_size,
      hidden_size = config.hidden_size,
      word_vec_size = config.word_vec_size,
      output_size = config.n_classes,
      n_layers = config.n_layers,
      dropout_p = config.dropout_p
  )
  optimizer = optim.Adam(model.parameters())
  crit = nn.NLLoss()
  print(model)

  if config.gpu >= 0:
    model.cuda(config.gpu)
    crit.cuda(config.gpu)

  rnn_trainer = T.Trainer(config)
  rnn_trainer = rnn_trainer.train(
      model,
      crit,
      optimizer,
      loaders.train_loader,
      loaders.valid_loader
  )

  torch.save({
      'rnn' : rnn_model.state_dict(),
      'config' : config,
      'vocab' : loaders.text.vocab,
      'classes' : loaders.lable.vocab,
  }, config.model_fn)

if __name__ == '__main__':
  config = define_argparser()
  main(config)