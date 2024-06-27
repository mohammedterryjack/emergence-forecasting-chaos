from jpype import JPackage

aisCalc3 = JPackage("infodynamics.measures.discrete").ActiveInformationCalculatorDiscrete

# mu = javaObject('infodynamics.utils.MatrixUtils');
# aisCalc3 = javaObject('infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete', base^phi_width, phi_K);
# aisCalc1 = javaObject('infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete', base, phi_K);

# for col=phi_width:width
#   aux = mu.computeCombinedValues(X(:,col-(phi_width-1):col), base);
#   aisCalc3.addObservations(aux);
# end
# aisCalc1.addObservations(X);

# % Then use test data to calculate Phi
# phi_locals  = zeros(size(test));
# ais3_locals = zeros(size(test));
# ais1_locals = zeros(size(test));

# test_aux = zeros(size(test));
# R = (phi_width-1)/2;
# for col=phi_width:testWidth
#   test_aux(:,col-1) = mu.computeCombinedValues(test(:,col-(phi_width-1):col), base);
# end
# ais3_locals = aisCalc3.computeLocalFromPreviousObservations(test_aux);
# ais1_locals = aisCalc1.computeLocalFromPreviousObservations(test);

# for col=1+R:testWidth-R
#   phi_locals(:,col) = ais3_locals(:,col) - sum(ais1_locals(:, col-R:col+R), 2);
# end

# end
