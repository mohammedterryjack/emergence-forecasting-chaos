public void initialise(){
		initialise(base, k, destEmbeddingDelay, sourceHistoryEmbedLength,
				sourceEmbeddingDelay, delay);
	}

/**
	 * Initialise with (potentially) new parameters
	 * 
	 * @param base
	 * @param destHistoryEmbedLength
	 * @param destEmbeddingDelay
	 * @param sourceHistoryEmbeddingLength
	 * @param sourceEmbeddingDelay
	 * @param delay
	 */
	public void initialise(int base, int destHistoryEmbedLength, int destEmbeddingDelay,
			int sourceHistoryEmbeddingLength, int sourceEmbeddingDelay, int delay) {
		
		boolean paramsChanged = (this.base != base) || (k != destHistoryEmbedLength) ||
				(this.destEmbeddingDelay != destEmbeddingDelay) || (this.sourceHistoryEmbedLength != sourceHistoryEmbeddingLength) ||
				(this.sourceEmbeddingDelay != sourceEmbeddingDelay) || (this.delay != delay); 
		super.initialise(base, destHistoryEmbedLength);
		if (paramsChanged) {
			updateParameters(base, destHistoryEmbedLength, destEmbeddingDelay,
				sourceHistoryEmbeddingLength, sourceEmbeddingDelay, delay);
		}
		
		if (paramsChanged || (sourceNextPastCount == null)) {
			// Create new storage for extra counts of observations
			try {
				sourceNextPastCount = new int[base_power_l][base][base_power_k];
				sourcePastCount = new int[base_power_l][base_power_k];
			} catch (OutOfMemoryError e) {
				// Allow any Exceptions to be thrown, but catch and wrap
				//  Error as a RuntimeException
				throw new RuntimeException("Requested memory for the base " +
						base + ", k=" + k + ", l=" + sourceHistoryEmbedLength +
						" is too large for the JVM at this time", e);
			}
		} else {
			MatrixUtils.fill(sourceNextPastCount, 0);
			MatrixUtils.fill(sourcePastCount, 0);
		}
		estimateComputed = false;
	}
	