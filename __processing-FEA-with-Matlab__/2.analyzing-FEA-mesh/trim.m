function [trimmed_data] = trim(const, sign, column, data)

% Gus Mueller - 12/16/2020

% trim: current revision of trim method for trimming XYZ data given a
% certain parameter in the X, Y, or Z data. 
%
% trim is a tool to quickly trim large data sets usign a simple system. The
% inputs are "const", "sign", "column", and "data". 
% - "const" (Integer) refers to the bound which you are trimming around
% - "sign" (> or <) refers to the direction of the trim 
% - "column" (Integer) refers to the column of data used to trim
% - "data" refers to the data used in the trim 
% The output is a fully trimmed data set
% Example:
% trim(150, >, 3, humerus_data) - This method will remove all data points
% that have a Z value below 150. 
trimmed_data = data;

if sign == "<"
trim_boolean = data(:,column) > const; 
trimmed_data(trim_boolean,:) = [];
elseif sign == ">"
trim_boolean = data(:,column) < const; 
trimmed_data(trim_boolean,:) = [];
end

end
