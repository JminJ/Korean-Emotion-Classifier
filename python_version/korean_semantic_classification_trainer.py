from copy import copy
import numpy as numpy
import torch
from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from korean_semactic_classification_utils as utils

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 1

class MyEngine(Engine):
    def __init__(self, func, model, crit, optimizer, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func)

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine, mini_batch):
        engine.model.train()
        engine.optimizer.zero_grad()

        x, y = mini_batch.text, mini_batch.label
        x, y = x.to(engine.device), y.to(engine.device)

        x = x[:, :engine.config.max_length]

        y_hat = engine.model(x)

        loss = engine.crit(y_hat, y)
        loss.backward()

        if isinstance(y, torch.longTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim = -1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0

        p_norm = float(utils.get_parameter_norm(engine.model.parameters()))
        g_norm = float(utils.get_grad_norm(engine.model.parameters()))

        engine.optimizer.step()

        return{
            'loss' : float(loss),
            'accuracy' : float(accuracy),
            '|param|' : p_norm,
            '|g_norm|' : g_norm,
        }
    
    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch.text, mini_batch.label
            x, y = x.to(engnie.device), y.to(engine.device)

            x = x[:, :engnie.config.max_length]

            y_hat = engine.model(x)

            loss = engine.crit(y_hat, y)

            if instance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim = -1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

            return {
                'loss' : float(loss),
                'accuracy' : float(accuracy),
            }
    
    @staticmethod
    def attach(train_engine, validation_engine, verbose = VERBOSE_BATCH_WISE):
        def attatch_running_average(engine, meric_name):
            RunningAverage(output_transform = lambda x : x[metric_name]).attach(
                engine,
                meric_name,
            )

            training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|']

            for metric_name in training_metric_names:
                attatch_running_average(train_engine, metric_name)

            if verbose >= VERBOSE_BATCH_WISE:
                pbar = ProgressBar(bar_format = None, ncols = 120)
                pbar.attach(train_engine, training_metric_names)

            if verbose >= VERBOSE_EPOCH_WISE:
                @train_engine.on(Events.EPOCH_COMPLETED)
                def print_train_logs(engine):
                    print('Epoch {} - |param\ = {:.2e} loss = {:.4e} accuracy = {}').format(
                        engine.state.epoch,
                        engine.state.metrics['|param|'],
                        engnie.state.metrics['|g_param|'],
                        engine.state.metrics['loss'],
                        engnie.state.metrics['accuracy']
                    )
                validation_metric_names = ['loss','accuracy']

                for metric_name in validation_metric_names:
                    attatch_running_average(validation_engine, metric_name)

                if verbose >= VERBOSE_EPOCH_WISE:
                    @validation_engine.on(Events.EPOCH_COMPLETED)
                    def print_valid_logs(engine):
                        print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:4.}'.format(
                            engine.state.metrics['loss'],
                            engine.state.metrics['accuracy'],
                            engine.best_loss,
                        ))

                    @staticmethos
