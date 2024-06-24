/**
	 * Computes local apparent transfer entropy for the given
	 *  states, using PDFs built up from observations previously
	 *  sent in via the addObservations method.
	 *  
 	 * @param source source time-series
	 * @param dest destination time-series. 
	 *  Must be same length as source
	 * @return time-series of local TE values
	 */
	public double[] computeLocalFromPreviousObservations(int source[], int dest[]){
		int timeSteps = dest.length;

		// Allocate for all rows even though we'll leave the first ones as zeros
		double[] localTE = new double[timeSteps];
		average = 0;
		max = 0;
		min = 0;

		if (timeSteps - startObservationTime <= 0) {
			// No observations to compute locals for
			return localTE;
		}
		
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
				pastVal[d] += dest[startObservationTime + d - 1
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
				sourcePastVal[d] += source[startObservationTime + d - delay
				                    - (sourceHistoryEmbedLength-1)*sourceEmbeddingDelay
				                    + p*sourceEmbeddingDelay];
				sourcePastVal[d] *= base;
			}
		}
		
		// now compute the local values
		int destVal, destEmbeddingPhase = 0, sourceEmbeddingPhase = 0;
		double logTerm;
		for (int t = startObservationTime; t < timeSteps; t++) {
			// First update the embedding values for the current
			//  phases of the embeddings:
			if (k > 0) {
				pastVal[destEmbeddingPhase] += dest[t-1];
			}
			sourcePastVal[sourceEmbeddingPhase] += source[t-delay];
			destVal = dest[t];
			int thisPastVal = pastVal[destEmbeddingPhase];
			int thisSourceVal = sourcePastVal[sourceEmbeddingPhase];
			// Now compute the local value
			logTerm = ((double) sourceNextPastCount[thisSourceVal][destVal][thisPastVal] /
							(double) sourcePastCount[thisSourceVal][thisPastVal]) /
			 		((double) nextPastCount[destVal][thisPastVal] / (double) pastCount[thisPastVal]);
			localTE[t] = Math.log(logTerm) / log_2;
			average += localTE[t];
			if (localTE[t] > max) {
				max = localTE[t];
			} else if (localTE[t] < min) {
				min = localTE[t];
			}
			// Now, update the combined embedding values and phases,
			//  for this phase we back out the oldest value which we'll no longer need:
			if (k > 0) {
				pastVal[destEmbeddingPhase] -= maxShiftedValue[dest[t-1-(k-1)*destEmbeddingDelay]];
				pastVal[destEmbeddingPhase] *= base; // and shift the others up
			}
			sourcePastVal[sourceEmbeddingPhase] -=
					maxShiftedSourceValue[
					    source[t-delay-(sourceHistoryEmbedLength-1)*sourceEmbeddingDelay]];
			sourcePastVal[sourceEmbeddingPhase] *= base; // and shift the others up
			// then update the phase
			destEmbeddingPhase = (destEmbeddingPhase + 1) % destEmbeddingDelay; 
			sourceEmbeddingPhase = (sourceEmbeddingPhase + 1) % sourceEmbeddingDelay; 
		}

		average = average/(double) (timeSteps - startObservationTime);
		
		return localTE;
	}