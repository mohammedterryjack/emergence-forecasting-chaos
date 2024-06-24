function LocalPhi()

phi54 = ComputePhi(54);
figure;
imshow(-phi54, 'Colormap', coolwarm, 'InitialMagnification', 300);
caxis([-3,3]);

phi110 = ComputePhi(110);
figure;
imshow(-phi110, 'Colormap', coolwarm, 'InitialMagnification', 300);
caxis([-5,5]);

end


function [ phi_locals ] = ComputePhi(rule)

%% Set parameter values
% How much of a point's neighbourhood to consider when calculating Phi
phi_width = 3;

% Slightly different embedding parameters for each rule. These were determined
% with a previous embedding analysis
if rule == 54
  phi_K = 4;
elseif rule == 110
  phi_K = 5;
else
  error('Only rules 54 and 110 supported.');
end

% ECA simulation parameters extracted from Lizier's thesis, section 3.3.
width = 10000;
T = 600;
transient = 30;

K = 8;
J = 1;
base = 2;

% Random number seed, for reproducibility
rng(9);


%% Simulate training data and generate testing data
X = elementaryCellularAutomata(rule, T + transient, 1.0*(rand([width,1]) < 0.5)');

if rule == 54
  testWidth = 67;
  test = elementaryCellularAutomata(rule, testWidth, 1.0*(rand([testWidth,1]) < 0.5)');

else
  initial_110 = [
          0,0,0,1,0,0,1,1,0,1,1,1,1,1, ...
          0,0,0,1,1,1,0,1,1,1, ...
          0,0,0,1,0,0,1,1,0,1,1,1,1,1, ...
          0,0,0,1,0,0,1,1,0,1,1,1, ...
          1,0,0,1,1,1,1, ...
          0,0,0,1,0,0,1,1,0,1,1,1,1,1, ...
          0,0,0,1,0,0,1,1,0,1,1,1,1,1
          ];
  testWidth = length(initial_110);
  test = elementaryCellularAutomata(110, 100, initial_110);
end

%% Calculate integrated information
% Embedded atomic Phi using AIS calculators. We use the atomic partition to
% calculate Phi (i.e. splitting all parts of the system)

% First add the training data to build the probability distributions
mu = javaObject('infodynamics.utils.MatrixUtils');
aisCalc3 = javaObject('infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete', base^phi_width, phi_K);
aisCalc1 = javaObject('infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete', base, phi_K);

for col=phi_width:width
  aux = mu.computeCombinedValues(X(:,col-(phi_width-1):col), base);
  aisCalc3.addObservations(aux);
end
aisCalc1.addObservations(X);

% Then use test data to calculate Phi
phi_locals  = zeros(size(test));
ais3_locals = zeros(size(test));
ais1_locals = zeros(size(test));

test_aux = zeros(size(test));
R = (phi_width-1)/2;
for col=phi_width:testWidth
  test_aux(:,col-1) = mu.computeCombinedValues(test(:,col-(phi_width-1):col), base);
end
ais3_locals = aisCalc3.computeLocalFromPreviousObservations(test_aux);
ais1_locals = aisCalc1.computeLocalFromPreviousObservations(test);

for col=1+R:testWidth-R
  phi_locals(:,col) = ais3_locals(:,col) - sum(ais1_locals(:, col-R:col+R), 2);
end

end

