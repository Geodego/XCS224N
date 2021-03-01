% loads housing data for further processing
%
% dramage 20070921

data = dlmread('portland.csv');

prices = data(:,1)/1000;
sizes  = data(:,2);
rooms  = data(:,3);

