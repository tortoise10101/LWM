import numpy as np
import torch


class ReplayBuffer(object):
    '''
    ReplayBufferをdataloader的な形式で再利用
    一連の文章の終わりの文を，プッシュするときは done=True
    observation is vector of sequence of text
    '''
    def __init__(self, capacity, observation_shape):
        self.capacity = capacity

        self.observations = np.zeros(
            (capacity, observation_shape))
        #    (capacity, observation_shape), dtype=np.uint8)
        self.done = np.zeros((capacity, 1), dtype=np.bool)

        self.index = 0
        self.is_filled = False

    def push(self, observation, done):
        self.observations[self.index] = observation
        self.done[self.index] = done

        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, chunk_length):
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []
        for _ in range(batch_size):
            cross_border = True
            while cross_border:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(
                    initial_index <= episode_borders,
                    episode_borders < final_index).any()
            sampled_indexes += list(range(initial_index, final_index + 1))

        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, -1)
        #    batch_size, chunk_length, *self.observations.shape[1:])
        sampled_done = self.done[sampled_indexes].reshape(
            batch_size, chunk_length, 1)

        return sampled_observations, sampled_done

    def __len__(self):
        return self.capacity if self.is_filled else self.index
