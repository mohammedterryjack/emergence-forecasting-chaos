class MyClass:
    def __init__(self):
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

    def initialise(self, base, destHistoryEmbedLength, destEmbeddingDelay,
                   sourceHistoryEmbeddingLength, sourceEmbeddingDelay, delay):
        paramsChanged = (self.base != base) or (self.k != destHistoryEmbedLength) or \
                        (self.destEmbeddingDelay != destEmbeddingDelay) or \
                        (self.sourceHistoryEmbedLength != sourceHistoryEmbeddingLength) or \
                        (self.sourceEmbeddingDelay != sourceEmbeddingDelay) or \
                        (self.delay != delay)
        
        # Assuming super.initialise(base, destHistoryEmbedLength) does some initialization
        # Here you should call the parent class initialise method if there is one
        
        if paramsChanged:
            self.updateParameters(base, destHistoryEmbedLength, destEmbeddingDelay,
                                  sourceHistoryEmbeddingLength, sourceEmbeddingDelay, delay)

        if paramsChanged or (self.sourceNextPastCount is None):
            try:
                self.sourceNextPastCount = [[[0 for _ in range(self.base_power_k)] for _ in range(base)] for _ in range(self.base_power_l)]
                self.sourcePastCount = [[0 for _ in range(self.base_power_k)] for _ in range(self.base_power_l)]
            except MemoryError as e:
                raise RuntimeError(f"Requested memory for the base {base}, k={self.k}, l={sourceHistoryEmbeddingLength} is too large for the system at this time") from e
        else:
            self.fill(self.sourceNextPastCount, 0)
            self.fill(self.sourcePastCount, 0)

        self.estimateComputed = False

    def updateParameters(self, base, destHistoryEmbedLength, destEmbeddingDelay,
                         sourceHistoryEmbeddingLength, sourceEmbeddingDelay, delay):
        # Implement the logic for updating the parameters here
        self.base = base
        self.k = destHistoryEmbedLength
        self.destEmbeddingDelay = destEmbeddingDelay
        self.sourceHistoryEmbedLength = sourceHistoryEmbeddingLength
        self.sourceEmbeddingDelay = sourceEmbeddingDelay
        self.delay = delay

    def fill(self, matrix, value):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if isinstance(matrix[i][j], list):
                    self.fill(matrix[i][j], value)
                else:
                    matrix[i][j] = value

# Example usage
my_obj = MyClass()
my_obj.initialise(10, 5, 3, 7, 2, 4)
