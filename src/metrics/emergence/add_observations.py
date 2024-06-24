class TransferEntropy:
    def __init__(self, start_observation_time, dest_embedding_delay, k, base, source_embedding_delay,
                 source_history_embed_length, delay, source_next_past_count, source_past_count,
                 next_past_count, past_count, next_count, max_shifted_value, max_shifted_source_value):
        self.start_observation_time = start_observation_time
        self.dest_embedding_delay = dest_embedding_delay
        self.k = k
        self.base = base
        self.source_embedding_delay = source_embedding_delay
        self.source_history_embed_length = source_history_embed_length
        self.delay = delay
        self.source_next_past_count = source_next_past_count
        self.source_past_count = source_past_count
        self.next_past_count = next_past_count
        self.past_count = past_count
        self.next_count = next_count
        self.max_shifted_value = max_shifted_value
        self.max_shifted_source_value = max_shifted_source_value
        self.observations = 0

    def add_observations(self, source, dest, start_time, end_time):
        if (end_time - start_time) - self.start_observation_time + 1 <= 0:
            # No observations to add
            return
        if end_time >= len(dest) or end_time >= len(source):
            raise IndexError(f"endTime ({end_time}) must be <= length of input arrays (dest: {len(dest)}, source: {len(source)})")
        
        # Increment the count of observations
        self.observations += (end_time - start_time) - self.start_observation_time + 1

        # Initialize and store the current previous values;
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
                past_val[d] += dest[start_time + self.start_observation_time + d - 1
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
                source_past_val[d] += source[start_time + self.start_observation_time + d - self.delay
                                             - (self.source_history_embed_length - 1) * self.source_embedding_delay
                                             + p * self.source_embedding_delay]
                source_past_val[d] *= self.base

        # 1. Count the tuples observed
        dest_val = 0
        dest_embedding_phase = 0
        source_embedding_phase = 0

        for r in range(start_time + self.start_observation_time, end_time + 1):
            # First update the embedding values for the current
            #  phases of the embeddings:
            if self.k > 0:
                past_val[dest_embedding_phase] += dest[r - 1]
            source_past_val[source_embedding_phase] += source[r - self.delay]
            # Add to the count for this particular transition:
            dest_val = dest[r]
            this_past_val = past_val[dest_embedding_phase]
            this_source_val = source_past_val[source_embedding_phase]
            self.source_next_past_count[this_source_val][dest_val][this_past_val] += 1
            self.source_past_count[this_source_val][this_past_val] += 1
            self.next_past_count[dest_val][this_past_val] += 1
            self.past_count[this_past_val] += 1
            self.next_count[dest_val] += 1

            # Now, update the combined embedding values and phases,
            #  for this phase we back out the oldest value which we'll no longer need:
            if self.k > 0:
                past_val[dest_embedding_phase] -= self.max_shifted_value[dest[r - 1 - (self.k - 1) * self.dest_embedding_delay]]
                past_val[dest_embedding_phase] *= self.base  # and shift the others up
            source_past_val[source_embedding_phase] -= \
                self.max_shifted_source_value[
                    source[r - self.delay - (self.source_history_embed_length - 1) * self.source_embedding_delay]]
            source_past_val[source_embedding_phase] *= self.base  # and shift the others up

            # then update the phase
            dest_embedding_phase = (dest_embedding_phase + 1) % self.dest_embedding_delay
            source_embedding_phase = (source_embedding_phase + 1) % self.source_embedding_delay
