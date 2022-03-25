# copyright @ 2019 shentianxiao
# copy from https://github.com/shentianxiao/text-autoencoders
import torch

from .dae import DAE
from .utils import loss_kl
from .vocab import Vocab


class VAE(DAE):
    """Variational Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)

    @classmethod
    def load(cls, path='./seq2vec/trained/model.pt'):
        dt = torch.load(path)
        vocab = Vocab('./seq2vec/trained/vocab.txt')
        model = VAE(vocab, dt['args'])
        model.load_state_dict(dt['model'])
        return model

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_kl * losses['kl']

    def autoenc(self, inputs, targets, is_train=False):
        mu, logvar, _, logits = self(inputs, is_train)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'kl': loss_kl(mu, logvar)}
