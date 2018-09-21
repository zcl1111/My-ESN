function plot_states(stateMatrix, states, nPoints, figNr)
  
% PLOT_STATES plots the internal states of the esn
%
% inputs:
% stateMatrix = matrix of size (nTrainingPoints) x
% (nInputUnits + nInternalUnits )
% stateMatrix(i,j) = internal activation of unit j after the 
% i-th training point has been presented to the network
% states = vector of size 1 x n , containing the indices of the internal
% units we want to plot
% nPoints = natural number containing the number of points to plot
% figNr: either [] or an integer. If [], a new figure is created, otherwise
% the plot is displayed in a figure window with number figNr
%
%
% example  : plot_states(stateMatrix,[1 2 3 4],200) plots the first 200
%points from the traces of the first 4 units



if isempty(figNr)    
    nFigure = figure ;
else
    nFigure = figNr;
end

figure(figNr); clf;

if (nargin < 3)
    nPoints = length(stateMatrix(:,1)) ; 
end

nStates = length(states) ; 

xMax = ceil(sqrt(nStates)) ; 
yMax = ceil(nStates /xMax);

for iPlot = 1 : nStates
    subplot(xMax,xMax,iPlot) ; 
    plot(stateMatrix(1:nPoints, states(1,iPlot)))
end

