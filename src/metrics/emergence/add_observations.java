	public void addObservations(int[] source, int[] dest) {
		addObservations(source, dest, 0, dest.length-1);
	}

/**
 	 * Add observations for a single source-destination pair
 	 *  to our estimates of the pdfs.
	 * Start and end time are the (inclusive) indices within which to add the observations.
	 * The start time is from the earliest of the k historical values of the destination (inclusive),
	 *  the end time is the last destination time point to add in.
	 *  
	 * @param source source time-series
	 * @param dest destination time-series. 
	 *  Must be same length as source
	 * @param startTime earliest time that we may extract embedded history from
	 * @param endTime last destination (next) time point to add in
	 * 
	 */
	public void addObservations(int[] source, int[] dest, int startTime, int endTime) {
		if ((endTime - startTime) - startObservationTime + 1 <= 0) {
			// No observations to add
			return;
		}
		if ((endTime >= dest.length) || (endTime >= source.length)) {
			throw new ArrayIndexOutOfBoundsException(
					String.format("endTime (%d) must be <= length of input arrays (dest: %d, source: %d)",
							endTime, dest.length, source.length));
		}
		// increment the count of observations:
		observations += (endTime - startTime) - startObservationTime + 1; 
		
		// Initialise and store the current previous values;
		//  one for each phase of the embedding delay.
		// First for the destination:
		int[] pastVal = new int[destEmbeddingDelay];
		for (int d = 0; d < destEmbeddingDelay; d++) {
			// Compute the current previous values for
			//  phase d of the embedding delay, but leave
			//  out the most recent value (we'll add those in
			//  in the main loop)
			pastVal[d] = 0;
			for (int p = 0; p < k-1; p++) {
				pastVal[d] += dest[startTime + startObservationTime + d - 1
				                   - (k-1)*destEmbeddingDelay
				                   + p*destEmbeddingDelay];
				pastVal[d] *= base;
			}
		}
		// Next for the source:
		int[] sourcePastVal = new int[sourceEmbeddingDelay];
		for (int d = 0; d < sourceEmbeddingDelay; d++) {
			// Compute the current previous values for
			//  phase d of the embedding delay, but leave
			//  out the most recent value (we'll add those in
			//  in the main loop)
			sourcePastVal[d] = 0;
			for (int p = 0; p < sourceHistoryEmbedLength - 1; p++) {
				sourcePastVal[d] += source[startTime + startObservationTime + d - delay
				                    - (sourceHistoryEmbedLength-1)*sourceEmbeddingDelay
				                    + p*sourceEmbeddingDelay];
				sourcePastVal[d] *= base;
			}
		}
		
		// 1. Count the tuples observed
		int destVal, destEmbeddingPhase = 0, sourceEmbeddingPhase = 0;
		for (int r = startTime + startObservationTime; r <= endTime; r++) {
			// First update the embedding values for the current
			//  phases of the embeddings:
			if (k > 0) {
				pastVal[destEmbeddingPhase] += dest[r-1];
			}
			sourcePastVal[sourceEmbeddingPhase] += source[r-delay];
			// Add to the count for this particular transition:
			// (cell's assigned as above)
			destVal = dest[r];
			int thisPastVal = pastVal[destEmbeddingPhase];
			int thisSourceVal = sourcePastVal[sourceEmbeddingPhase];
			sourceNextPastCount[thisSourceVal][destVal][thisPastVal]++;
			sourcePastCount[thisSourceVal][thisPastVal]++;
			nextPastCount[destVal][thisPastVal]++;
			pastCount[thisPastVal]++;
			nextCount[destVal]++;
			// Now, update the combined embedding values and phases,
			//  for this phase we back out the oldest value which we'll no longer need:
			if (k > 0) {
				pastVal[destEmbeddingPhase] -= maxShiftedValue[dest[r-1-(k-1)*destEmbeddingDelay]];
				pastVal[destEmbeddingPhase] *= base; // and shift the others up
			}
			sourcePastVal[sourceEmbeddingPhase] -=
					maxShiftedSourceValue[
					    source[r-delay-(sourceHistoryEmbedLength-1)*sourceEmbeddingDelay]];
			sourcePastVal[sourceEmbeddingPhase] *= base; // and shift the others up
			// then update the phase
			destEmbeddingPhase = (destEmbeddingPhase + 1) % destEmbeddingDelay; 
			sourceEmbeddingPhase = (sourceEmbeddingPhase + 1) % sourceEmbeddingDelay; 
		}
	}