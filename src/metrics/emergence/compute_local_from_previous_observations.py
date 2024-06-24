import numpy as np

class TransferEntropy:
    def __init__(self, start_observation_time, dest_embedding_delay, k, base, source_embedding_delay,
                 source_history_embed_length, delay, log_2, source_next_past_count, source_past_count,
                 next_past_count, past_count, max_shifted_value, max_shifted_source_value):
        self.start_observation_time = start_observation_time
        self.dest_embedding_delay = dest_embedding_delay
        self.k = k
        self.base = base
        self.source_embedding_delay = source_embedding_delay
        self.source_history_embed_length = source_history_embed_length
        self.delay = delay
        self.log_2 = log_2
        self.source_next_past_count = source_next_past_count
        self.source_past_count = source_past_count
        self.next_past_count = next_past_count
        self.past_count = past_count
        self.max_shifted_value = max_shifted_value
        self.max_shifted_source_value = max_shifted_source_value
        self.average = 0
        self.max = 0
        self.min = 0

    def compute_local_from_previous_observations(self, source, dest):
        time_steps = len(dest)

        # Allocate for all rows even though we'll leave the first ones as zeros
        local_te = np.zeros(time_steps)
        self.average = 0
        self.max = float('-inf')
        self.min = float('inf')

        if time_steps - self.start_observation_time <= 0:
            # No observations to compute locals for
            return local_te

        # Initialise and store the current previous values;
        #  one for each phase of the embedding delay.
        # First for the destination:
        past_val = np.zeros(self.dest_embedding_delay, dtype=int)
        for d in range(self.dest_embedding_delay):
            # Compute the current previous values for
            #  phase d of the embedding delay, but leave
            #  out the most recent value (we'll add those in
            #  in the main loop)
            past_val[d] = 0
            for p in range(self.k - 1):
                past_val[d] += dest[self.start_observation_time + d - 1
                                    - (self.k - 1) * self.dest_embedding_delay
                                    + p * self.dest_embedding_delay]
                past_val[d] *= self.base

        # Next for the source:
        source_past_val = np.zeros(self.source_embedding_delay, dtype=int)
        for d in range(self.source_embedding_delay):
            # Compute the current previous values for
            #  phase d of the embedding delay, but leave
            #  out the most recent value (we'll add those in
            #  in the main loop)
            source_past_val[d] = 0
            for p in range(self.source_history_embed_length - 1):
                source_past_val[d] += source[self.start_observation_time + d - self.delay
                                             - (self.source_history_embed_length - 1) * self.source_embedding_delay
                                             + p * self.source_embedding_delay]
                source_past_val[d] *= self.base

        # now compute the local values
        dest_val = 0
        dest_embedding_phase = 0
        source_embedding_phase = 0

        for t in range(self.start_observation_time, time_steps):
            # First update the embedding values for the current
            #  phases of the embeddings:
            if self.k > 0:
                past_val[dest_embedding_phase] += dest[t - 1]
            source_past_val[source_embedding_phase] += source[t - self.delay]
            dest_val = dest[t]
            this_past_val = past_val[dest_embedding_phase]
            this_source_val = source_past_val[source_embedding_phase]

            # Now compute the local value
            log_term = (self.source_next_past_count[this_source_val][dest_val][this_past_val] /
                        self.source_past_count[this_source_val][this_past_val]) / \
                       (self.next_past_count[dest_val][this_past_val] /
                        self.past_count[this_past_val])
            local_te[t] = np.log(log_term) / self.log_2
            self.average += local_te[t]
            if local_te[t] > self.max:
                self.max = local_te[t]
            elif local_te[t] < self.min:
                self.min = local_te[t]

            # Now, update the combined embedding values and phases,
            #  for this phase we back out the oldest value which we'll no longer need:
            if self.k > 0:
                past_val[dest_embedding_phase] -= self.max_shifted_value[dest[t - 1 - (self.k - 1) * self.dest_embedding_delay]]
                past_val[dest_embedding_phase] *= self.base  # and shift the others up
            source_past_val[source_embedding_phase] -= \
                self.max_shifted_source_value[
                    source[t - self.delay - (self.source_history_embed_length - 1) * self.source_embedding_delay]]
            source_past_val[source_embedding_phase] *= self.base  # and shift the others up

            # then update the phase
            dest_embedding_phase = (dest_embedding_phase + 1) % self.dest_embedding_delay
            source_embedding_phase = (source_embedding_phase + 1) % self.source_embedding_delay

        self.average /= (time_steps - self.start_observation_time)

        return local_te
