# copyright @ 2019 shentianxiao
# copy from https://github.com/shentianxiao/text-autoencoders

from .dae import DAE
from .utils import loss_kl


class VAE(DAE):
    """Variational Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_kl * losses['kl']

    def autoenc(self, inputs, targets, is_train=False):
        mu, logvar, _, logits = self(inputs, is_train)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'kl': loss_kl(mu, logvar)}
