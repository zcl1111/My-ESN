function esn = generate_esn2(nInputUnits, nInternalUnits, nOutputUnits, varargin)
% Creates an ESN set up for use in multiple-channel output association tasks. 
% The number of input, internal, and output 
% units have to be set. Any other option is set using the format 
% 'name_of_options1',value1,'name_of_option2',value2, etc.
% 
%%%%% input arguments:
% nInputUnits: the dimension of the input 
% nInternalUnits: size of the Esn
% nOutputUnits: the dimension of the output
%
%%%%% optional arguments:
% 'inputScaling': a nInputUnits x 1 vector
%
% 'inputShift': a nInputUnits x 1 vector. 
%
% 'teacherScaling': a nOutputUnits x 1 vector
%
% 'teacherShift': a nOutputUnits x 1 vector. 
%
% 'noiseLevel': a small number containing the amount of uniform noise to be
%  added when computing the internal states
%
% 'learningMode': a string ('offline_singleTimeSeries', 'offline_multipleTimeSeries' or 'online')
%     1. Case 'offline_singleTimeSeries': trainInput and trainOutput each represent a 
%        single time series in an array of size sequenceLength x sequenceDimension
%     2. Case 'offline_multipleTimeSeries': trainInput and trainOutput each represent a 
%        collection of K time series, given in cell arrays of size K x 1, where each cell is an
%        array of size individualSequenceLength x sequenceDimension
%     3. Case 'online': trainInput and trainOutput are a single time
%        series, output weights are adapted online
%
% 'reservoirActivationFunction': a string ("tanh", "identity", "sigmoid01") ,
%
% 'outputActivationFunction': a string("tanh", "identity", "sigmoid01") ,
%
% 'inverseOutputActivationFunction': the inverse to
%    outputActivationFunction, one of 'atanh', 'identity', 'sigmoid01_inv'.
%    When choosing the activation function, make sure the inverse
%    activation function is corectly set.
%
% 'methodWeightCompute': a string ('pseudoinverse', 'wiener_hopf'). It  
%    specifies which method to use to compute the output weights given the
%    state collection matrix and the teacher
%
% 'spectralRadius': a positive number less than 1. 
%
% 'feedbackScaling': a nOutputUnits x 1 vector, indicating the scaling
%     factor to be applied on the output before it is fed back into the network
%
% 'type': a string ('plain_esn', 'leaky_esn' or 'twi_esn')
% 'trained': a flag indicating whether the network has been trained already
% 'timeConstants': option used in networks with type == "leaky_esn", "leaky1_esn" and "twi_esn".
%                      Is given as column vector of size esn.nInternalUnitsm, where each entry 
%                      signifies a time constant for a reservoir neuron.
% 'leakage': option used in networks with type == "leaky_esn" or "twi_esn"
% 'RLS_lambda': option used in online training(learningMode == "online") 
% 'RLS_delta': option used in online training(learningMode == "online")
%

%%����Ϊwiener_hopf�����淽�̼��㣬���߶����м���

%%%% set the number of units
esn.nInternalUnits = nInternalUnits; 
esn.nInputUnits = nInputUnits; 
esn.nOutputUnits = nOutputUnits; 
  
connectivity = min([10/nInternalUnits 1]);
nTotalUnits = nInternalUnits + nInputUnits + nOutputUnits; 

esn.internalWeights_UnitSR = generate_internal_weights(nInternalUnits, ...
                                                connectivity);
esn.nTotalUnits = nTotalUnits; 

%rand( 'seed', 42 ); 

% input weight matrix has weight vectors per input unit in colums
esn.inputWeights = 2.0 * rand(nInternalUnits, nInputUnits)- 1.0;

% output weight matrix has weights for output units in rows
% includes weights for input-to-output connections
esn.outputWeights = zeros(nOutputUnits, nInternalUnits + nInputUnits);

%output feedback weight matrix has weights in columns
esn.feedbackWeights = (2.0 * rand(nInternalUnits, nOutputUnits)- 1.0);

%{init default parameters

esn.inputScaling  = ones(nInputUnits, 1);
esn.inputShift    = zeros(nInputUnits, 1);
esn.teacherScaling= ones(nOutputUnits, 1);
esn.teacherShift  = zeros(nOutputUnits, 1);
esn.noiseLevel = 0.0 ; 
esn.reservoirActivationFunction = 'tanh';%���뼤������Ϊ���к���
esn.outputActivationFunction = 'identity' ; % options: identity or tanh or sigmoid01�����������Ϊ����
esn.methodWeightCompute = 'wiener_hopf' ; % options: pseudoinverse and wiener_hopfα�淨����ع鷨
esn.inverseOutputActivationFunction = 'identity' ; 
esn.spectralRadius = 1 ; %�װ뾶Ϊ1
esn.feedbackScaling = zeros(nOutputUnits, 1); %��������Ϊ0
esn.trained = 0 ; 
esn.type = 'plain_esn' ; 
esn.timeConstants = ones(esn.nInternalUnits,1); 
esn.leakage = 0.5;  
esn.learningMode = 'offline_multipleTimeSeries' ; 
esn.RLS_lambda = 1 ; 


args = varargin; 
nargs= length(args);
for i=1:2:nargs
  switch args{i},
   case 'inputScaling', esn.inputScaling = args{i+1} ; 
   case 'inputShift', esn.inputShift= args{i+1} ; 
   case 'teacherScaling', esn.teacherScaling = args{i+1} ; 
   case 'teacherShift', esn.teacherShift = args{i+1} ;     
   case 'noiseLevel', esn.noiseLevel = args{i+1} ; 
   case 'learningMode', esn.learningMode = args{i+1} ; 
   case 'reservoirActivationFunction',esn.reservoirActivationFunction=args{i+1};
   case 'outputActivationFunction',esn.outputActivationFunction=  ...
                        args{i+1};        
   case 'inverseOutputActivationFunction', esn.inverseOutputActivationFunction=args{i+1}; 
   case 'methodWeightCompute', esn.methodWeightCompute = args{i+1} ; 
   case 'spectralRadius', esn.spectralRadius = args{i+1} ;  
   case 'feedbackScaling',  esn.feedbackScaling = args{i+1} ; 
   case 'type' , esn.type = args{i+1} ; 
   case 'timeConstants' , esn.timeConstants = args{i+1} ; 
   case 'leakage' , esn.leakage = args{i+1} ; 
   case 'RLS_lambda' , esn.RLS_lambda = args{i+1};
   case 'RLS_delta' , esn.RLS_delta = args{i+1};
       
      otherwise
          error('the option does not exist');
  end 
end


%%%% error checking
% check that inputScaling has correct format
if length(esn.inputScaling(:,1)) ~= esn.nInputUnits
    error('the size of the inputScaling does not match the number of input units'); 
end
if length(esn.inputScaling(1,:)) ~= 1
    error('inputScaling should be provided as a column vector of size nInputUnits x 1'); 
end
% check that inputShift has correct format
if length(esn.inputShift(:,1)) ~= esn.nInputUnits
    error('the size of the inputScaling does not match the number of input units'); 
end
if length(esn.inputShift(1,:)) ~= 1
    error('inputScaling should be provided as a column vector of size nInputUnits x 1'); 
end

if length(esn.teacherScaling(:,1)) ~= esn.nOutputUnits
    error('the size of the teacherScaling does not match the number of output units'); 
end
if length(esn.teacherScaling(1,:)) ~= 1
    error('teacherScaling should be provided as a column vector of size nOutputUnits x 1'); 
end

if length(esn.teacherShift(:,1)) ~= esn.nOutputUnits
    error('the size of the teacherShift does not match the number of output units'); 
end
if length(esn.teacherShift(1,:)) ~= 1
    error('teacherShift should be provided as a column vector of size nOutputUnits x 1'); 
end
if length(esn.timeConstants) ~= esn.nInternalUnits
    error('timeConstants must be given as column vector of length esn.nInternalUnits'); 
end
if ~strcmp(esn.learningMode,'offline_singleTimeSeries') &&...
        ~strcmp(esn.learningMode,'offline_multipleTimeSeries') && ...
        ~strcmp(esn.learningMode,'online')
    error('learningMode should be either "offline_singleTimeSeries", "offline_multipleTimeSeries" or "online" ') ; 
end
if ~((strcmp(esn.outputActivationFunction,'identity') && ...
        strcmp(esn.inverseOutputActivationFunction,'identity')) || ...
        (strcmp(esn.outputActivationFunction,'tanh') && ...
        strcmp(esn.inverseOutputActivationFunction,'atanh')) || ...
        (strcmp(esn.outputActivationFunction,'sigmoid01') && ...
        strcmp(esn.inverseOutputActivationFunction,'sigmoid01_inv')))  ...

    error('outputActivationFunction and inverseOutputActivationFunction do not match'); 
end


if 0
if strcmp(esn.type,'leaky_esn') || strcmp(esn.type,'twi_esn')
    checkMat = esn.internalWeights_UnitSR + esn.retainment * eye(esn.nInternalUnits);
    opts.disp = 0 ; 
    effective_specRad = max(abs(eigs(checkMat,1,'LM',opts)));
    if effective_specRad > 1
        disp('Warning: a leaky integrator ESN has been created with effective SR exceeding 1');
    end
end
end
