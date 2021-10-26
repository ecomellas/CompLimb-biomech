function [axispoint,axisvec,radius,rmse] = cylinderfit(xyz)
% cylinderfit: fits a 3-d cylinder to data
% usage: [axispoint,axisvec,radius] = cylinderfit(XYZ)
%
% CYLINDERFIT is a regression modeling tool, but not a true
% total least squares fit for the cylinder parameters. That
% would be far more computationally intensive than I think is
% necessary. Instead, this tool employs a variable partitioning
% process. If the axis of the cylinder was known, then it would
% be possible to project the points into the plane perpendicular
% to that cylinder. In that plane, the points will now lie in a
% nice circle, for which the parameters are easily estimated
% using a distinct tool designed to fit a circle to data. Thus
% an optimization is performed over the unknown axis of the
% cylinder, with a circle fit being done internally.
%
% arguments: (input)
%  xyz - nx3 array of data points. Each point is one row of xyz.
%        n must be at least 6.
%
% arguments: (output)
%  axispoint - 1x3 vector that denotes a point along the axis of
%        the cylinder.
%
%  axisvec - 1x3 vector that denotes a vector that points along
%        the cylinder axis. axisvec will be normalized to have
%        unit 2-norm.
%
%  radius - radius of the cylinder
%
%  rmse - the final measure used for quality of fit
%
% Example:
% % Generate data that falls on a cylinder
% n = 50;
% theta = rand(n,1)*2*pi;
% z = rand(n,1);
% x = cos(theta)*2;y = sin(theta)*2;
% xyz = [x,y,z] + randn(n,3)/10;
%
% % The original radius was 2. The cylinder is oriented
% % along the z axis, so axisvec should be approximately
% % [0 0 1]. The standard deviation of the additive
% % error was 0.1. All of these numbers are well estimated
% % given the noise.
%
% [axispoint,axisvec,radius,rmse] = cylinderfit(xyz)
%
% % axispoint =
% %      0.029324     0.011413      0.45987
% % axisvec =
% %     -0.055504    0.0075907      0.99843
% % radius =
% %        2.0068
% % rmse =
% %       0.10328
%
% see also: circlefit
%
% Author: John D'Errico
% e-mail: woodchips@rochester.rr.com
% Release: 1.0
% Release date: 7/21/2011

% defaults, error checks
[n,dim] = size(xyz);
if dim~=3
  error('xyz must be an nx3 array of data.')
elseif n<=6
  error('xyz must have at least 6 rows. (At least 6 data points.)')
end

% transform xyz to have 0 mean. this might
% improve some of the numerics later on
xyzbar = mean(xyz,1);
xyzt = xyz - repmat(xyzbar,n,1);

% A good starting point for the rotation may come from
% the svd of the data. For example, if the data truly lies
% on a long, thin cylinder, then the vector corresponding to
% the largest singular value will point along the axis of the
% cylinder. Likewise, if the data lies on a short, squat
% cylinder, then the smallest singular value will indicate
% the axis of the cylinder. In either case, the other two
% singular values will be roughly equal to each other.
[V,S] = eig(xyzt'*xyzt); %#ok
bestrmse = inf;
startangles = [0 0];
for i = 1:3
  % test each right singular vector as a rotation.
  rotationangles = recoverRotationAngles(V(:,i));
  
  % do the corresponding circle fit
  rmse = cylinderObj(rotationangles);
  
  if rmse < bestrmse
    % this one is better than what we have tried, so keep it
    startangles = rotationangles;
    bestrmse = rmse;
  end
end

% choose an optimizer. Use fminunc if it is there.
% otherwise, use fminsearch.
if exist('fminunc','file') == 2
  % we can use fminunc
  opts = optimset('fminunc');
  opts.Display = 'off';
  opts.LargeScale = 'off';
  
  [finalrotationangles,fval,exitflag] = fminunc(@cylinderObj,startangles,opts);
  
else
  % no choice but to use fminsearch
  opts = optimset('fminsearch');
  opts.Display = 'off';
  
  [finalrotationangles,fval,exitflag] = fminsearch(@cylinderObj,startangles,opts);
  
end

% do one final call to the objective function to recover the
% estimated radius and final choice of axisvec
[rmse,axispoint,axisvec,radius] = cylinderObj(finalrotationangles);

% translate the final axispoint back, recovering the initial
% translation.
axispoint = axispoint + xyzbar;

% ===========================================
% end main line, begin nested functions
% ===========================================

  function [rmse,axispoint,axisvec,radius] = cylinderObj(rotationangles)
    % nested optimizer objective function. The objective value
    % to optimize itself will be the computed rmse from circlefit.
    
    % compute the axis vector, as well as the rotation
    % matrix into a 2-d plane.
    [axisvec,nullvecs] = rotationMatrix(rotationangles);
    
    % project the 3-d data into a 2-d domain.
    xy = xyzt*nullvecs';
    
    % and do the circlefit
    [C,radius,rmse] = circlefit(xy);
    
    % project the circle center back into the 3-d domain
    axispoint = C*nullvecs;
    
  end

end

% ===========================================
% end nested functions, begin sub functions
% ===========================================

function [axisvec,nullvecs] = rotationMatrix(rotationangles)
% compute a 3x3 rotation matrix from two angles
% theta is the rotation around the y axis (so [0,pi] radians)
% phi is rotation around the z axis (also [0,pi] radians)
theta = rotationangles(1);
phi = rotationangles(2);

R = [cos(phi) sin(phi) 0;-sin(phi) cos(phi) 0;0 0 1];
R = R * [cos(theta) 0 sin(theta);0 1 0;-sin(theta) 0 cos(theta)];
% axisvec is the transformed unit vector [1 0 0]
axisvec = R(1,:);
nullvecs = R(2:3,:);
end

% ===========================================

function rotationangles = recoverRotationAngles(axisvec)
% function to recover phi & theta from the cylinder axial vector
rotationangles(2) = asin(axisvec(2));
rotationangles(1) = acos(axisvec(1)/cos(rotationangles(2)));
end

% ===========================================




