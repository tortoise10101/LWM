import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class TransitionModel(nn.Module):
    def __init__(
        self, state_dim, rnn_hidden_dim,
            hidden_dim=30, min_stddev=0.1):
        super(TransitionModel, self).__init__()
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim, hidden_dim)

        self.fc_state_mean_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_prior = nn.Linear(hidden_dim, state_dim)

        self.fc_rnn_hidden_embedded_obs = nn.Linear(
            rnn_hidden_dim + 128, hidden_dim)

        self.fc_state_mean_posterior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_posterior = nn.Linear(hidden_dim, state_dim)

        self.rnn = nn.GRUCell(
            input_size=state_dim,
            hidden_size=rnn_hidden_dim)
        self._min_stddev = min_stddev

    def recurrent(self, state, rnn_hidden):
        """
        h_{t+1} = f(h_t, s_t)
        """
        return self.rnn(state, rnn_hidden)

    def prior(self, rnn_hidden):
        """
        prior p(s_{t+1} | h_{t+1})
        """
        hidden = self.fc_rnn_hidden(rnn_hidden)

        mean = self.fc_state_mean_prior(hidden)
        stddev = F.softplus(
            self.fc_state_stddev_prior(hidden)) + self._min_stddev

        return Normal(mean, stddev), rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs):
        """
        posterior q(s_{t+1}| h_{t+1}, e_{t+1})
        """
        hidden = self.fc_rnn_hidden_embedded_obs(
            torch.cat([rnn_hidden, embedded_obs], dim=1))

        mean = self.fc_state_mean_posterior(hidden)
        stddev = F.softplus(
            self.fc_state_stddev_posterior(hidden)) + self._min_stddev

        return Normal(mean, stddev)

    def forward(self, state, rnn_hidden, embedded_next_obs):
        next_state_prior, rnn_hidden = \
            self.prior(self.recurrent(state, rnn_hidden))
        next_state_posterior = \
            self.posterior(rnn_hidden, embedded_next_obs)

        return next_state_prior, next_state_posterior, rnn_hidden


class ObservationModel(nn.Module):
    """
    p(o_t| s_t, h_t)
    o_t: vector of text-sequence
    """
    def __init__(self, state_dim, rnn_hidden_dim):
        super(ObservationModel, self).__init__()
        self.fc = nn.Linear(state_dim + rnn_hidden_dim, 128)

    def forward(self, state, rnn_hidden):
        obs = self.fc(torch.cat([state, rnn_hidden], dim=1))
        return obs


class RSSM:
    def __init__(self, state_dim, rnn_hidden_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transition = TransitionModel(state_dim, rnn_hidden_dim).to(self.device)
        self.observation = ObservationModel(state_dim, rnn_hidden_dim).to(self.device)