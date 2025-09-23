import numpy as np

class Policy:
    def __init__(self, chromosome, obs_size, action_size, hidden_size=8):
        # Longitudes
        in_hidden = obs_size * hidden_size
        hidden_out = hidden_size * action_size
        total_len = in_hidden + hidden_out

        assert len(chromosome) == total_len, (
            f"Chromosome length {len(chromosome)} does not match "
            f"expected {total_len} for obs={obs_size}, hidden={hidden_size}, actions={action_size}"
        )

        # Reconstruir pesos
        self.w1 = chromosome[:in_hidden].reshape(obs_size, hidden_size)
        self.w2 = chromosome[in_hidden:].reshape(hidden_size, action_size)

    def act(self, obs):
        h = np.maximum(0, obs @ self.w1)   # ReLU
        logits = h @ self.w2
        return np.argmax(logits)