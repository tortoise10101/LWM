# copyright @ 2019 shentianxiao
# copy from https://github.com/shentianxiao/text-autoencoders
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import reparameterize, log_prob, noisy


class TextModel(nn.Module):
    """Container module with word embedding and projection layers"""

    def __init__(self, vocab, args, initrange=0.1):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.embed = nn.Embedding(vocab.size, args.dim_emb)
        self.proj = nn.Linear(args.dim_h, vocab.size)

        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)

    def tokenize(self, text):
        if isinstance(text, str):
            return self.__tokenize(text)
        # batch
        seqs = [i.split() for i in text]
        max_len = max([len(s) for s in seqs])
        go_x = []
        for s in seqs:
            s_idx = [self.vocab.go] + \
                [self.vocab.word2idx[w] if w in self.vocab.word2idx else self.vocab.unk for w in s]
            padding = [self.vocab.pad] * (max_len-len(s))
            go_x.append(s_idx + padding)
        return torch.LongTensor(go_x).t().contiguous()

    def __tokenize(self, text):
        text = text.split()
        s_idx = [self.vocab.go] + \
            [self.vocab.word2idx[w] if w in self.vocab.word2idx else self.vocab.unk for w in text]
        return torch.tensor(s_idx).view(-1, 1)

    def detokenize(self, v):
        sents = []
        for s in v.t():
            sents.append([self.vocab.idx2word[id] for id in s[1:]])
        return self.strip_eos(sents)

    def strip_eos(self, sents):
        return [sent[:sent.index('<eos>')] \
            if '<eos>' in sent else sent for sent in sents]


class DAE(TextModel):
    """Denoising Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.drop = nn.Dropout(args.dropout)
        self.E = nn.LSTM(
            input_size=args.dim_emb,
            hidden_size=args.dim_h,
            num_layers=args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0,
            bidirectional=True)
        self.G = nn.LSTM(
            input_size=args.dim_emb,
            hidden_size=args.dim_h,
            num_layers=args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0)

        self.h2mu = nn.Linear(args.dim_h*2, args.dim_z)
        self.h2logvar = nn.Linear(args.dim_h*2, args.dim_z)
        self.z2emb = nn.Linear(args.dim_z, args.dim_emb)
        self.opt = optim.Adam(
            self.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def flatten(self):
        self.E.flatten_parameters()
        self.G.flatten_parameters()

    def encode(self, input):
        input = self.drop(self.embed(input))
        _, (h, _) = self.E(input)
        print(h.shape)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.h2mu(h), self.h2logvar(h)

    def decode(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def generate(self, z, max_len, alg):
        assert alg in ['greedy', 'sample', 'top5']
        sents = []
        input = torch.zeros(
            1, len(z), dtype=torch.long, device=z.device).fill_(self.vocab.go)
        hidden = None
        for _ in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg == 'sample':
                input = torch.multinomial(
                    logits.squeeze(dim=0).exp(), num_samples=1).t()
            elif alg == 'top5':
                not_top5_indices = logits.topk(
                    logits.shape[-1]-5, dim=2, largest=False).indices
                logits_exp = logits.exp()
                logits_exp[:, :, not_top5_indices] = 0.
                input = torch.multinomial(
                    logits_exp.squeeze(dim=0), num_samples=1).t()
        return torch.cat(sents)

    def forward(self, input, is_train=False):
        _input = noisy(
            self.vocab, input, *self.args.noise) if is_train else input
        mu, logvar = self.encode(_input)
        z = reparameterize(mu, logvar)
        logits, _ = self.decode(z, input)
        return mu, logvar, z, logits

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def loss(self, losses):
        return losses['rec']

    def autoenc(self, inputs, targets, is_train=False):
        _, _, _, logits = self(inputs, is_train)
        return {'rec': self.loss_rec(logits, targets).mean()}

    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        # `clip_grad_norm` helps prevent the exploding
        # gradient problem in RNNs / LSTMs.
        # nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.opt.step()

    def nll_is(self, inputs, targets, m):
        """compute negative log-likelihood by importance sampling:
           p(x;theta) = E_{q(z|x;phi)}[p(z)p(x|z;theta)/q(z|x;phi)]
        """
        mu, logvar = self.encode(inputs)
        tmp = []
        for _ in range(m):
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, inputs)
            v = log_prob(z, torch.zeros_like(z), torch.zeros_like(z)) - \
                self.loss_rec(logits, targets) - log_prob(z, mu, logvar)
            tmp.append(v.unsqueeze(-1))
        ll_is = torch.logsumexp(torch.cat(tmp, 1), 1) - np.log(m)
        return -ll_is
