function [C,R,rmse] = circlefit(data,solverflag)
% circlefit: fits a circle (or a sphere in higher dimensions) to data in n dimensions
% usage: [C,R,rmse] = circlefit(data,robustflag)
%
% Note: this code fits a circle to data that is assumed to lie
% on the perimeter of the circle. In n-dimensions with n > 2
% this code fits a sphere or hyper-sphere. This is NOT a bounding
% circle/sphere code, but a regression modeling tool.
%
% If you have data that lies in a circle in a higher dimensional
% space, then you way wish to use a projection into the plane of the
% circle.
%
% arguments: (input)
%  data - must be an n by p array, where the
%       dimension of the data is p (p>=2) and
%       n is the number of data points.
%
%       We must have n strictly greater than
%       p, so you cannot fit a sphere in 3
%       dimensions with only 3 data points.
%       When n = p + 1, the sphere will pass
%       exactly through the data points unless
%       the data points are degenerate.
%
%  solverflag - character - 
%       A flag that denotes if robustfit from
%       the stats toolbox will be used, or if
%       backslash is acceptable, or if pinv
%       should be used. 
%
%       '\' or 'backslash' --> Use backslash (\)
%       'robustfit' --> use the function robustfit from the stats TB
%       'pinv' --> Use pinv
%
%       Default: '\'
%
% arguments: (output)
%  C  - a 1xp vector, that contains the center
%       of the sphere in the R^p dimensional
%       space of the data.
%
%  R  - (scalar) radius of the sphere as
%       estimated
%
%  rmse - Root mean square of the residual errors
%       over all of the points, in terms of their
%       deviations from a circle.
%
% Example:
% % Data on the surface of a 5-sphere. Center should be [1 2 3 4 5],
% % with a nominal radius of 2.
%
%  data = randn(100,5);
%  data = data./repmat(sqrt(sum(data.^2,2)),1,5);
%  data = bsxfun(@plus,[1 2 3 4 5],2*data) + randn(size(data))/50);
%
%  [C,R,rmse] = circlefit(data);
%
% C =
%     0.99761       1.9972       3.0022        3.999       5.0077
%
% R =
%     2.0013
%
% rmse =
%     0.020022
%
%
% See also: minboundcircle, minboundsphere, cylinderfit
%
%
% Author: John D'Errico
% e-mail: woodchips@rochester.rr.com
% Release: 1.0
% Release date: 12/12/2010

% test the argument
if (nargin < 1) || (nargin > 2)
  error('Accepts only one or two arguments')
end

if nargin == 1
  solverflag = 0;
elseif ~ismember(solverflag,[0 1])
  error('If supplied, robustflag must be 0 or 1')
end

[n,p] = size(data);
if p < 2
  error('The data must lie in a p-dimensional space, where p >= 2')
elseif (n <= p)
  error('There must be at least p+1 data points (rows of data)')
end

% n must be at least 3 from the prior tests.
% the idea here is we can find the center of the circle/sphere
% by subtracting the equation for one of the points from
% another. This neatly drops out the quadratic terms in the
% unknowns, as well as the radius term. It leaves us with only
% linear terms in the unknowns, so therefore a linear system
% solve will be adequate to solve for the center point. The
% problem is, we still have some squared terms there in the
% data coefficients. So if there is noise in the data, then
% the squared terms will add a bias to the solution. A logical
% solution seems to be to subtract various points from each
% other to minimize the bias seen. For large values of n (the
% number of data points) I get tricky in how I choose what to
% subtract.
if n == 3
  I2 = [3 1 2 2 3 1]';
  I1 = repmat((1:n)',2,1);
  k = 2;
elseif n <= 5
  I2 = (1:n)';
  I2 = [circshift(I2,1);circshift(I2,2);circshift(I2,3)];
  I1 = repmat((1:n)',3,1);
  k = 3;
else
  % trying to be tricky here...
  coprimes = primes(100);
  coprimes = setdiff(coprimes,unique(factor(n)));
  I2 = 1 + mod((1:n)'*coprimes(1),n);
  I2 = [I2;1 + mod(I2*coprimes(2),n);1+mod(circshift(I2,1)*coprimes(3),n)];
  I1 = repmat((1:n)',3,1);
  k = 3;
end

% build the linear least squares problem for
% the center coordinates of the sphere.
A = zeros(k*n,p);
rhs = zeros(k*n,1);
for j = 1:p
  A(:,j) = 2*(data(I1,j) - data(I2,j));
  rhs = rhs + data(I1,j).^2 - data(I2,j).^2;
end

% solve using backslash for the center
if solverflag
  C = (A\rhs).';
else
  C = robustfit(A,rhs,[],[],'off')';
end

% recover the radius
R = sqrt(mean(sum((data - repmat(C,n,1)).^2,2)));

% a measure of the errors, in terms of the distance from each
% point to the estimated circle
resids = sqrt(sum((data - repmat(C,n,1)).^2,2)) - R;
rmse = sqrt(mean(resids.^2));


