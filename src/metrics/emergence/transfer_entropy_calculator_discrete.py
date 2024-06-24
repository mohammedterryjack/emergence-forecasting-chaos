class TransferEntropyDiscrete:
    def __init__(self) -> None:
        self.base = None
        self.destHistoryEmbedLength = None
        self.destEmbeddingDelay = None
        self.sourceHistoryEmbedLength = None
        self.sourceEmbeddingDelay = None
        self.delay = None
        self.sourceNextPastCount = None
        self.sourcePastCount = None
        self.estimateComputed = False
        self.base_power_l = 0
        self.base_power_k = 0
        self.k = 0

        self.sourceNextPastCount = [[[0 for _ in range(self.base_power_k)] for _ in range(base)] for _ in range(self.base_power_l)]
        self.sourcePastCount = [[0 for _ in range(self.base_power_k)] for _ in range(self.base_power_l)]

        self.estimateComputed = False

    def add_observation(self, x:list[int], y:list[int]) -> None:
        pass 

    def compute_local_from_previous_observations(self, x:list[int], y:list[int]) -> list[int]:
        pass 
