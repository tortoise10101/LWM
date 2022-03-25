from os import stat
import torch
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from worldmodel import RSSM, ReplayBuffer
from seq2vec import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load trained seq2vec model
seq2vec = VAE.load()

latent_dim = 128
state_dim = 30
rnn_hidden_dim = 200
rssm = RSSM(
    state_dim=state_dim,
    rnn_hidden_dim=rnn_hidden_dim)

model_lr = 6e-4
eps = 1e-4
model_params = (
    list(rssm.transition.parameters()) +
    list(rssm.observation.parameters()))
model_optimizer = torch.optim.Adam(model_params, lr=model_lr, eps=eps)

cap = 100000
replay_buffer = ReplayBuffer(capacity=cap, observation_shape=latent_dim)

gmmma = 0.9
lam = 0.95
free_nats = 3
clip_grad_norm = 100

batch_size = 10
chunk_length = 20


next_state_posterior = None  # TODO
def train():
    global seq2vec, rssm, model_optimizer, replay_buffer
    global next_state_posterior  # TODO
    for t in range(1000):
        observations, _ = replay_buffer.sample(batch_size, chunk_length)
        observations = torch.as_tensor(observations, device=device)
        observations = observations.view(chunk_length, batch_size, -1)
        # observations = observations.transpose(3, 4).transpose(2, 3)
        # observations = observations.transpose(0, 1)

        states = torch.zeros(
            chunk_length, batch_size, state_dim, device=device)
        rnn_hiddens = torch.zeros(
            chunk_length, batch_size, rnn_hidden_dim, device=device)

        state = torch.zeros(batch_size, state_dim, device=device)
        rnn_hidden = torch.zeros(batch_size, rnn_hidden_dim, device=device)

        kl_loss = 0
        for l in range(chunk_length-1):
            next_state_prior, next_state_posterior, rnn_hidden = \
                rssm.transition(state, rnn_hidden, observations[l+1])
            state = next_state_posterior.rsample()
            states[l+1] = state
            rnn_hiddens[l+1] = rnn_hidden
            kl = kl_divergence(
                next_state_prior, next_state_posterior).sum(dim=1)
            kl_loss += kl.clamp(min=free_nats).mean()
            # kl_loss += kl.mean()
        kl_loss /= (chunk_length-1)

        states = states[1:]
        rnn_hiddens = rnn_hiddens[1:]

        flatten_states = states.view(-1, state_dim)
        flatten_rnn_hiddens = rnn_hiddens.view(-1, rnn_hidden_dim)
        recon_observations = \
            rssm.observation(
                flatten_states,
                flatten_rnn_hiddens
                ).view(chunk_length-1, batch_size, latent_dim)

        obs_loss = \
            0.5 * F.mse_loss(
                recon_observations.float(),
                observations[1:].float(), reduction='none').mean([0, 1]).sum()

        model_loss = kl_loss + obs_loss
        model_optimizer.zero_grad()
        model_loss.backward()
        clip_grad_norm_(model_params, clip_grad_norm)
        model_optimizer.step()

        print('update_step: %3d model loss: %.5f, kl_loss: %.5f, obs_loss: %.5f' \
            % (t, model_loss.item(), kl_loss.item(), obs_loss.item()))


def prepare_buffer():
    global replay_buffer
    # with open('dataset/wiki.train.raw', 'r') as f:
    with open('dataset/wiki.valid.raw', 'r') as f:
        r = f.readlines()
        # for s in f.readlines():
        bs = 100
        for i in range(0, len(r)-bs, bs):
            print(100*i/len(r), '%')
            if i/len(r) > 0.1:
                break
            _, _, z, _ = seq2vec.forward(
                seq2vec.tokenize(r[i:i+bs]))
            z = z.detach().numpy()
            for j in z:
                replay_buffer.push(j, False)


def load_buffer():
    global replay_buffer
    import pickle
    with open('dataset/wtest.pickle', 'rb') as f:
        replay_buffer = pickle.load(f)


# prepare_buffer()
load_buffer()

obs, _ = replay_buffer.sample(batch_size, chunk_length)
obs = torch.as_tensor(obs, device=device)
obs = obs.view(chunk_length, batch_size, -1)

train()