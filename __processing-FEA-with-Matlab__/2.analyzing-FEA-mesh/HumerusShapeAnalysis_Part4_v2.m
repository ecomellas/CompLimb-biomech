%% HumerusShapeAnalysis
% HumerusShapeAnalysis is a tool designed to quantify the 
% characteristic epicondyles ("bumps") of the axolotl humerus. 
% Part 1: Alignment of the limb based on the axis of the ulna bone rudiment.
% Part 2: Computation of heatmaps and metrics to evaluate the 3D shape of the humerus.
% Part 3: Evaluation of the 3D shape of control vs mechano-sensitivity-impaired limbs.
% Part 4: 17 dpa cavitation limb & mesh comparison.
%
%% Part 4: 17 dpa cavitation limb & mesh comparison
%
% This script is based on the pipeline that analyzes 3D shape from 
% experimental data. It will read in the original mesh surface and the
% segmented 17dpa humerus and compute their surface maps.
%
%% Authors
% Ester Comellas
% e.comellas@northeastern.edu
% Northeastern University, May 2021
%
%% User instructions
% NOTE: This code has been developed and tested using a macOS system.
% Users using a different OS might need to adapt some parts of the code,
% e.g. slashes in filepath might need to be "\" instead of "/".
%
% Save in a folder named "input" the following file:
% + FEA surface (.vtk file containing FEA mesh, see below for instructions)
% + cavitation limb point cloud (.mat file with point clouds of bone rudiment surfaces)
% + mean heatmaps (.mat file with heatmaps)
%
% The following user-generated codes are required to run this script:
% + trim.m 
% + circlefit.m
% + cylinderfit.m
%
% Define the desired parameters for the analysis in the section below.
% 
%% Instructions to process FEA results
% No radius or ulna pointclouds expected. Humerus is expected to be
% correctly aligned following same criteria as experimental limbs.
% Humerus heatmap will be computed like for experimental data.
%
% To obtain coordinates of surface points in Paraview:
% 1) Open deal.ii surface results (bcs.pvd) and in the Properties tab, 
% deactivate all solutions by unticking the boxes.
% 2) Save as .vtk by going to File > Save Data:
%  ! Select "Legacy VTK files(*.vtk)". 
%  ! Write the file name with .vtk extension. 
%  ! Make sure to tick the option "write timesteps as file-series" 
%    and File type ASCII. 
% 3) Delete all files generated except the first one. 
%    This is the file to be read by this Matlab script: 
%     - "POINTS" are the coordinates (xyz) of the displacement nodes.
%     - "CELLS" are the faces based on the displacement nodes. 
% 
% Add file to input folder for processing in this script.
%
%% Some initial actions and global definitions
close all; %close all current windows

runprob = {'stl_cav';...
           'FEA_cav_coarse'};
new_filenames = {'Cavitation17dpaLimb.mat';...
                 'FEA_cav_coarse.vtk'};
plotmkr = { 's';...
            '*';...
            '*';...
            '*'};
            
% Define name for results files. 
filename = 'FinalResults.mat';

%% User data
% Define factor for fitted radius (= Grid scale used in deal.ii)
grid_scale =  2.5;

% Define if first cylinder fit must be used for subsequent analysis (=1),
% or each analysis should be fitted separately (=0).
cyl_fit = 0;

%Define num. of refinements to upsample the point clouds of the FEA results
max_refine = 10;

% Define height of cylindrical part in normalized bullet shape
norm_cylinder_height = 3.0;

% Define bin_width to compute continuous heatmaps from scatter plot 
% There is a lower limit (approx 0.03). For values too small, the resulting 
% arrays are too large for matlab's "histogram" functionto handle.
bin_width = 0.03; 

% For bump segmentation, how much bump is kept when trimming. Measure 
% taken from the center of the fitted hemisphere downwards, given as 
% percentage of fitted radius
bump_trim = 0.25; 

% Define colobar positive and negative limits for normalized heatmap
colorbar_limit = 0.65;

% Change to black if default background of point clouds is white in your
% Matlab version:
legend_text_color = 'white'; 

% Saving figures in matlab format takes a long time. 
% Activate (=1) or deactivate it here (=0).
savefigure = 0; 

% Make figures visible in matlab or save directly to file (visible figures
% are alse saved to file automatically)
visible = 'on';

% When loading .fig files saved by this script, if it doesn?t show, try 
% typing into the command line the following (immediately after loading it):
% f = gcf;
% set(f,'visible','on')

%% Check that input order is correct if cyl_fit == 1
if (cyl_fit==1 && contains(runprob{1},'stl') == 0 )
        error('First file to process MUST be stl cavitation.')
end

%% Create results folder and .mat file if they don't exist
if (exist('results', 'dir')~=7) 
    mkdir results;
end

% must load data from results mat file to avoid it being overwritten
% but if file doesn't exist, create it    
cd results/
if exist(filename,'file')==2
    load(filename);
else
    info = 'This file contains the data resulting from the humeri shape analysis. Including additional results like FEA predictions, heatmap means or cavitation limb.';
    save(filename,'info')
end
cd ..

% Define colors for the three bone rudiments
color_h = [211 211 211]; %RGB light gray
color_u = [ 65 105 225]; %RGB royal blue
color_r = [220  20  60]; %RGB crimson

% Define flat array size for continuous heatmaps based on user data
max_horiz = 2*pi; % Perimeter of the normalized bullet shape (radius=1)
max_vert = norm_cylinder_height + pi/2; % Height of cylinder + 1/2 perimeter of hemisphere
NumBins = [floor(max_horiz/bin_width), floor(max_vert/bin_width)];
clear max_horiz max_vert
stl_exists = 0;

% Loop through requested individual results
for probnum = 1:size(runprob,1)
    
    % Variables saved to mat file cannot contain dashes, only underscores
    varname = runprob{probnum};
    % Underscores in figure titles are interpreted as subscripts, use dashes
    titlename = replace(varname,'_','-');
    
    % Print info to Command Window
    X = sprintf('Running %s...',varname);
    disp(X)
    clear X
    
    %% Set new folder to save figures
    filepath = sprintf('results/%s',titlename);

    %If subfolder doesn't exist, create a new one
    if (exist(filepath, 'dir')~=7) 
        mkdir(filepath);
    end
    
    if contains(runprob{probnum},'stl') == 1
        %% Experimental data
        % load aligned limbs stored in .mat file
        cd input 
        load (new_filenames{probnum}); 
        cd ..

        % Extract names of all point clouds
        limb_list = fieldnames(BoneRudimentSurfaces);
        max_limb = size(limb_list,1)/3;

        % Check all limbs are complete (each limb must have 3 bone rudiments)
        if max_limb ~= 1
            error ('.mat file of cavitation limb must contain three bone rudiment pointclouds.')
        end
        
        % Extract surfaces of three bone rudiments for this limb
        ptCloud_humerus = BoneRudimentSurfaces.(limb_list{1}); 
        ptCloud_radius  = BoneRudimentSurfaces.(limb_list{2}); 
        ptCloud_ulna    = BoneRudimentSurfaces.(limb_list{3}); 
        
        % Point Cloud - imported point cloud
        figure('Name','Imported aligned elbow point cloud','visible',visible);
            pcshow(ptCloud_humerus)
            hold on 
            pcshow(ptCloud_radius)
            pcshow(ptCloud_ulna)
            hold off
            xlabel('X', 'FontSize', 16);
            ylabel('Y', 'FontSize', 16);
            zlabel('Z', 'FontSize', 16);
            axis equal;
            title (titlename, 'FontSize', 20);
            view(-50, 20)

        filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig1-aligned-limb.png'];
        saveas(gcf,filename_png);
        clear filename_png
        if savefigure==1
            filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig1-aligned-limb.fig'];
            savefig(filename_matlab);
            clear filename_matlab
        end
        
        % Center Humerus Around Origin for Analysis
        %Save coordinates of point clouds into arrays
        XYZ_humerus = ptCloud_humerus.Location;  
        XYZ_radius = ptCloud_radius.Location;                           
        XYZ_ulna = ptCloud_ulna.Location;    
        
        % Scale stl surface
        XYZ_humerus = XYZ_humerus*grid_scale;
        
        % Trimming off end of humerus as it is unimportant data and we 
        % need an "empty" bottom to be able to fit the hemisphere tangent to top
        % trim_value is computed as 20% of the height of the humerus
        trim_value = (max(XYZ_humerus(:,3))-min(XYZ_humerus(:,3)))*0.20;
        [XYZ_humerus] = trim(trim_value, '>', 3, XYZ_humerus); 
         
        % Translate coordinates to center all the data around the origin for 
        % further consistent analysis
        xyzcent_h_1 = [min(XYZ_humerus(:,1)), ...
                       min(XYZ_humerus(:,2)), ...
                       min(XYZ_humerus(:,3))];
        centxyz_h_1 = XYZ_humerus - xyzcent_h_1;
        centxyz_r_1 = XYZ_radius - xyzcent_h_1;
        centxyz_u_1 = XYZ_ulna - xyzcent_h_1;
        

        %% Fit Cylinder
        % Cylinder fit function for determining radius and fitting cylinder
        % (community generated) - not very efficient, takes a long time
        [cyl_center,axisvec,fitted_radius,~] = cylinderfit(centxyz_h_1);
        % Check that axis of fitted cylinder is approx vertical
        if (axisvec(1) > 0.05 || axisvec(2) > 0.05)
            disp('Careful: Axis of fitted cylinder might be too tilted. Check cylinder fit to aligned humerus is ok for this limb!');
        end
        clear axisvec
        
        %% Shift stl point cloud and cylinder to origin
        new_xyz_h_stl = centxyz_h_1;
        new_xyz_h_stl(:,1) = centxyz_h_1(:,1) - cyl_center(1);
        new_xyz_h_stl(:,2) = centxyz_h_1(:,2) - cyl_center(2);
        cyl_center(1) = 0;
        cyl_center(2) = 0;
        centxyz_h_1 = new_xyz_h_stl;
        
        % set the pointCloud for the new shifted data & assign color
        ptCloud_humerus_stl = pointCloud(centxyz_h_1);
        points_h = zeros(ptCloud_humerus_stl.Count,3);
        ptCloud_humerus_stl.Color = uint8(ones(size(points_h)).*color_h);

        clear points_h xyzcent_h_1 XYZ_humerus new_xyz_h_stl
        clear trim_value ptCloud_humerus
        
        %% Temporary Sphere Data and Manipulation
        % Finding Z of Cylinder center to offset the sphere center
        F = scatteredInterpolant(centxyz_h_1(:,1),...
                                 centxyz_h_1(:,2),...
                                 centxyz_h_1(:,3),'nearest','none') ;
        % where (xi,yi) is the location for which you want z
        z_test_find = F(cyl_center(1),cyl_center(2));   
        % this will be the center of the hemisphere above the cylinder

        clear F

        %% Identifying Center of Sphere and Axes
        % creating the scenter of the sphere to make sure the top of the sphere is
        % tangent with the top of the humerus. 
        sph_center = [cyl_center(1) , cyl_center(2), z_test_find - fitted_radius]; 

       
        %% Plotting Sphere and Cylinder - Axis and Tangency Point
        % Create cylinder data and reset height of cylinder
        [x_cyl, y_cyl, z_cyl] = cylinder(fitted_radius); 
        z_cyl_new = z_cyl*sph_center(3);
        clear z_cyl

        % Make unit sphere for hemisphere plot
        [x,y,z] = sphere(40);
        % Scale to desired radius from the cylinderfit method 
        x = x * fitted_radius;
        y = y * fitted_radius;
        z = z * fitted_radius;
        % Segmenting out the bottom part of the sphere for the visual, only keeping
        % the top hemisphere
        z(z < 0) = NaN;
        x(any(isnan(z), 2), :) = [];
        y(any(isnan(z), 2), :) = [];
        z(any(isnan(z), 2), :) = [];
        if ~isnan(centxyz_u_1)
            centroid_u_1 = [mean(centxyz_u_1(:,1)), mean(centxyz_u_1(:,2)), mean(centxyz_u_1(:,3))];
            centroid_r_1 = [mean(centxyz_r_1(:,1)), mean(centxyz_r_1(:,2)), mean(centxyz_r_1(:,3))];
        end    
        
        % figure to show the cylinder fitted to the humerus with different colors 
         figure('Name','"Bullet shape" fitted to the scaled humerus point cloud','visible',visible)
            q = surf(x+sph_center(1),y+sph_center(2),z+sph_center(3));
            set(q, 'FaceAlpha', 0.3,'FaceColor', 'y')
            hold on 
            pcshow(ptCloud_humerus_stl);
            plot3(cyl_center(1), cyl_center(2), z_test_find, '*g');
            plot3(cyl_center(1), cyl_center(2),  cyl_center(3), '*g');
            surf(x_cyl + cyl_center(1), y_cyl + cyl_center(2), z_cyl_new,...
                 'FaceAlpha', 0.3,'FaceColor', 'y')
            hold off
            axis equal;
            view(100,20)
            xlabel('X', 'FontSize', 16);
            ylabel('Y', 'FontSize', 16);
            zlabel('Z', 'FontSize', 16);
            title (titlename, 'FontSize', 20);

        filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig2-fitted-bullet-shape.png'];
        saveas(gcf,filename_png);
        clear filename_png
        if savefigure==1
            filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig2-fitted-bullet-shape.fig'];
            savefig(filename_matlab);
            clear filename_matlab
        end

        clear q
        
        stl_exists = 1;
    elseif contains(runprob{probnum},'FEA') == 1
    %% FEA data
    
        % Read in vtk file
        cd input
        fid = fopen(new_filenames{probnum},'r');
        cd ..
            
        input_mesh_str = textscan(fid,'%s','Delimiter','\r');
        fclose(fid);
        clear fid;
        
        cell_split = cellfun(@(x)regexp(x,' ','split'),input_mesh_str{1},...
                                 'UniformOutput',0);

        % Split into coordinates, elements and displacements
        node_pos = strfind(input_mesh_str{1},'POINTS');
        elem_pos = strfind(input_mesh_str{1},'CELLS');
        elem_type_pos = strfind(input_mesh_str{1},'CELL_TYPES');
        disp_pos = strfind(input_mesh_str{1},'displacement');
        
        k = 1;
        found = 1;
        while found
            if isempty(node_pos{k,1}) == 0
                idx_node = k;
                found = 0;
            end
            k = k + 1;
        end
        
        % counter k not reset because CELLS always come after POINTS
        found = 1;
        while found
            if isempty(elem_pos{k,1}) == 0
                idx_elem = k;
                found = 0;
            end
            k = k + 1;
        end
        
        % counter k not reset because CELL_YTYPE always come after POINTS
        found = 1;
        while found
            if isempty(elem_type_pos{k,1}) == 0
                idx_elem_type = k;
                found = 0;
            end
            k = k + 1;
        end
        
        % counter k not reset because displacement always come after
        % CELL_TYPE
        found = 1;
        while found
            if isempty(disp_pos{k,1}) == 0
                idx_disp = k;
                found = 0;
            end
            k = k + 1;
        end
        clear k found elem_pos disp_pos node_pos elem_type_pos
        
        idx_node_start = idx_node + 1;
        idx_node_end = idx_elem - 1;
        idx_elem_start = idx_elem + 1;
        idx_elem_end = idx_elem_type - 1;
        idx_disp_start = idx_disp + 1;
        idx_disp_end = size(input_mesh_str{1},1);
      
        cell_split_nodes = vertcat(cell_split{idx_node_start:idx_node_end});
        cell_split_elems = vertcat(cell_split{idx_elem_start:idx_elem_end});
        cell_split_disp = vertcat(cell_split{idx_disp_start:idx_disp_end});
        
        clear idx_node_start idx_node_end idx_node cell_split
        clear idx_elem_start idx_elem_end idx_elem idx_elem_type 
        clear idx_disp_start idx_disp_end idx_disp input_mesh_str
        
        if sum(size(cell_split_nodes) ~= size(cell_split_disp)) ~= 0
            error('Error reading vtk files. Node coordinates and node displacements donnot match.')
        end
        
        input_mesh_nodes = zeros(3*size(cell_split_nodes,1),3);
        input_mesh_disp = zeros(3*size(cell_split_nodes,1),3);
        m = 1;
        for k=1:size(cell_split_nodes,1)   
            for n=1:size(input_mesh_nodes,2)   
                input_mesh_nodes(m,n)   = str2double(cell_split_nodes{k,n});
                input_mesh_disp(m,n)   = str2double(cell_split_disp{k,n});
                input_mesh_nodes(m+1,n) = str2double(cell_split_nodes{k,n+3});
                input_mesh_disp(m+1,n) = str2double(cell_split_disp{k,n+3});
                input_mesh_nodes(m+2,n) = str2double(cell_split_nodes{k,n+6});
                input_mesh_disp(m+2,n) = str2double(cell_split_disp{k,n+6});
            end
            m = m + 3;
        end
        clear k n m cell_split_nodes cell_split_disp
        
        input_mesh_elems = zeros(size(cell_split_elems,1),4);
        for k=1:size(input_mesh_elems,1)   
            for n=1:size(input_mesh_elems,2)   
                input_mesh_elems(k,n) = str2double(cell_split_elems{k,n+1})+1;
                % Increment node number by one, so it starts at 1 not at 0.
            end
        end
        clear k n cell_split_elems
        
        % Create mesh structure based on the info extracted from vtk files
        mesh.vertices = input_mesh_nodes;
        mesh.faces = input_mesh_elems;
        clear input_mesh_nodes input_mesh_disp input_mesh_elems
            
        % plot
        title_txt = strcat(titlename, ' original mesh');
        figure('Name','Imported FEA mesh','visible',visible);
            patch(mesh,'FaceColor',       [0.92 0.92 0.92], ...
                       'FaceAlpha',       1,                ...
                       'EdgeColor',       'k',           ...
                       'FaceLighting',    'gouraud',        ...
                       'AmbientStrength', 0.15);
             camlight('headlight');
             material('dull');
             axis('image');
            view([-135 35]);
            xlabel('x (µm)');
            ylabel('y (µm)');
            zlabel('z (µm)');
            title (title_txt, 'FontSize', 20);
            grid on;
            axis equal;
             
        filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig0-1-imported-FEA-mesh.png'];
        saveas(gcf,filename_png);
        clear filename_png
        if savefigure==1
            filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig0-1-imported-FEA-mesh.fig'];
            savefig(filename_matlab);
            clear filename_matlab
        end
        clear title_txt
        
             
        % Upsample cloud point
        max_faces = size(mesh.faces,1);
        coord_per_elem = (4+max_refine)*max_refine;
        new_coord = NaN(max_faces*coord_per_elem,3);
        tot_count = 0;
        for e = 1:max_faces
            
            elem_nodes = mesh.faces(e,:);
            vertex = NaN(size(elem_nodes,2),3);
            for v = 1:size(elem_nodes,2)
                vertex(v,:) = mesh.vertices(elem_nodes(v),:);
            end
            clear v elem_nodes
            new_coord_elem = NaN(coord_per_elem,3);
            
            edge_1 = vertex(2,:)-vertex(1,:);
            edge_2 = vertex(3,:)-vertex(4,:);
            coord_edge_1 = NaN(max_refine+2,3);
            coord_edge_1(1,:) = vertex(1,:);
            coord_edge_1(end,:) = vertex(2,:);
            
            coord_edge_2 = NaN(max_refine+2,3);
            coord_edge_2(1,:) = vertex(4,:);
            coord_edge_2(end,:) = vertex(3,:);
            clear vertex
            
            for n = 1:max_refine
                coord_edge_1(1+n,:) = coord_edge_1(n,:) + ...
                                      edge_1/((max_refine+1)); %sqrt(sum(edge_1.^2))*
                coord_edge_2(1+n,:) = coord_edge_2(n,:) + ...
                                      edge_2/((max_refine+1));
            end
            new_coord_elem(1:max_refine,:) = coord_edge_1(2:1+max_refine,:);
            new_coord_elem(max_refine+1:2*max_refine,:) = coord_edge_2(2:1+max_refine,:);
            
            clear edge_1 edge_2 n
            
            for k = 1:(max_refine+2)
                edge_int = coord_edge_2(k,:) - coord_edge_1(k,:);
                coord_int = NaN(max_refine+2,3);
                coord_int(1,:) = coord_edge_1(k,:);
                coord_int(end,:) = coord_edge_2(k,:);
               	for n = 1:max_refine
                    coord_int(1+n,:) = coord_int(n,:) + ...
                                      edge_int/((max_refine+1));
                end
                new_coord_elem((k+1)*max_refine+1:(k+2)*max_refine,:) = coord_int(2:1+max_refine,:);
            end
            
            new_coord(tot_count+1:tot_count+coord_per_elem,:)=new_coord_elem;
            tot_count = tot_count + coord_per_elem;
            
            clear k new_coord_elem coord_edge_1 coord_edge_2 coord_int
            clear edge_int n
        end
        clear e pos_count coord_per_elem max_faces tot_count
        
        % Remove duplicated nodes
        new_coord_no_dupl = unique(new_coord,'rows');
        clear new_coord
        
        % Add initial coord to new ones
        total_coord = cat(1,mesh.vertices,new_coord_no_dupl); 
        
        % plot
        title_txt = strcat(titlename, ' upsampled points in mesh');
        figure('Name','Upsampled FEA mesh','visible',visible);
            patch(mesh,'FaceColor',       [0.92 0.92 0.92], ...
                       'FaceAlpha',       0.2,                ...
                       'EdgeColor',       'k',           ...
                       'FaceLighting',    'gouraud',        ...
                       'AmbientStrength', 0.15);
             hold on
             camlight('headlight');
             material('dull');
             axis('image');
             view([-135 35]);

             scatter3(new_coord_no_dupl(:,1),new_coord_no_dupl(:,2),new_coord_no_dupl(:,3),...
                         'SizeData',5,'MarkerFaceColor','red','MarkerEdgeColor','none');
             hold off
             xlabel('x (µm)');
             ylabel('y (µm)');
             zlabel('z (µm)');
             title (title_txt, 'FontSize', 20);
             grid on;
             axis equal;
             
        filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig0-2-uplsampled-FEA-mesh.png'];
        saveas(gcf,filename_png);
        clear filename_png
        if savefigure==1
            filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig0-2-uplsampled-FEA-mesh.fig'];
            savefig(filename_matlab);
            clear filename_matlab
        end
        clear title_txt 
        clear input_mesh_disp input_mesh_elems input_mesh_nodes new_coord_no_dupl
                    
        
        %% Create humerus point cloud and align to experimental stl
        % This is done manually and must be adjusted if the geometries are 
        % changed in the future
        % Rotate 180 deg around z axis
        ptCloud_humerus = pointCloud(total_coord*rotz(180));
        points_h = zeros(ptCloud_humerus.Count,3);
        ptCloud_humerus.Color = uint8(ones(size(points_h)).*color_h);
        clear points_h total_coord
        
        XYZ_humerus = ptCloud_humerus.Location;  
        xyzcent_h_1 = zeros(size(XYZ_humerus(:,1)));
            
        if stl_exists == 1
            % Translate coordinates to align with stl surface
    %         x_stl = (max(ptCloud_humerus_stl.Location(:,1)) - ...
    %                     min(ptCloud_humerus_stl.Location(:,1)) )/2;
    %         x_fea = (max(ptCloud_humerus.Location(:,1)) - ...
    %                     min(ptCloud_humerus.Location(:,1)) )/2;
    %         y_stl = (max(ptCloud_humerus_stl.Location(:,2)) - ...
    %                     min(ptCloud_humerus_stl.Location(:,2)) )/2;
    %         y_fea = (max(ptCloud_humerus.Location(:,2)) - ...
    %                     min(ptCloud_humerus.Location(:,2)) )/2;
            z_shift =  z_test_find;
    %         xyzcent_h_1 (:,1) = x_stl - x_fea;
    %         xyzcent_h_1 (:,2) = y_stl - y_fea;
            xyzcent_h_1 (:,3) = z_shift - max(XYZ_humerus(:,3));
            clear x_stl x_fea y_stl y_fea z_shift
        else
            % Translate to positive octant
            xyzcent_h_1 (:,3) = - min(XYZ_humerus(:,3));
        end
        
        centxyz_h_1 = XYZ_humerus + xyzcent_h_1;
        centxyz_r_1 = NaN;
        centxyz_u_1 = NaN;
        
        % Trimming off end of humerus as it is unimportant data and we 
        % need an "empty" bottom to be able to fit the hemisphere tangent to top
        trim_value = (max(centxyz_h_1(:,3))-min(centxyz_h_1(:,3)))*0.02;
        [centxyz_h_1] = trim(trim_value, '>', 3, centxyz_h_1); 
        
        % set the pointClouds for the new rotated data & assign colors
        ptCloud_humerus_fea = pointCloud(centxyz_h_1);
    
        points_h = zeros(ptCloud_humerus_fea.Count,3);
        ptCloud_humerus_fea.Color = uint8(ones(size(points_h)).*color_h);
        
        mesh.vertices = centxyz_h_1;

        
        if stl_exists == 1
            figure('Name','Aligned FEA mesh and stl surface','visible',visible);
               patch(mesh,'FaceColor',       [0.92 0.92 0.92], ...
                           'FaceAlpha',       0.5,                ...
                           'EdgeColor',       'k',           ...
                           'FaceLighting',    'gouraud',        ...
                           'AmbientStrength', 0.15);
                camlight('headlight');
                material('dull');
                axis('image');
                hold on
                pcshow(ptCloud_humerus_stl,'MarkerSize',0.1)
                view([-135 35]);
                xlabel('x (µm)');
                ylabel('y (µm)');
                zlabel('z (µm)');
                title (titlename, 'FontSize', 20);
                grid on;
                axis equal;

            filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig1-FEA-humerus-and-stl.png'];
            saveas(gcf,filename_png);
            clear filename_png
            if savefigure==1
                filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig1-FEA-humerus-and-stl.fig'];
                savefig(filename_matlab);
                clear filename_matlab
            end 
        end
        
        figure('Name','Aligned FEA mesh','visible',visible);
            pcshow(ptCloud_humerus_fea,'MarkerSize',0.1)
            view([-135 35]);
            xlabel('x (µm)');
            ylabel('y (µm)');
            zlabel('z (µm)');
            title (titlename, 'FontSize', 20);
            grid on;
            axis equal;

        filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig1-FEA-humerus-points.png'];
        saveas(gcf,filename_png);
        clear filename_png
        if savefigure==1
            filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig1-FEA-humerus-points.fig'];
            savefig(filename_matlab);
            clear filename_matlab
        end 
        
        clear points_h points_u points_r
        clear xyzcent_h_1 trim_value XYZ_humerus %XYZ_radius XYZ_ulna 
        clear ptCloud_humerus ptCloud_ulna ptCloud_radius
         
        if cyl_fit == 0
            %% Fit Cylinder
            % Cylinder fit function for determining radius and fitting cylinder
            % (community generated) - not very efficient, takes a long time
            [cyl_center,axisvec,fitted_radius,~] = cylinderfit(centxyz_h_1);
            % Check that axis of fitted cylinder is approx vertical
            if (axisvec(1) > 0.05 || axisvec(2) > 0.05)
                disp('Careful: Axis of fitted cylinder might be too tilted. Check cylinder fit to aligned humerus is ok for this limb!');
            end
            clear axisvec

            %% Shift stl point cloud and cylinder to origin
            new_xyz_h_stl = centxyz_h_1;
            new_xyz_h_stl(:,1) = centxyz_h_1(:,1) - cyl_center(1);
            new_xyz_h_stl(:,2) = centxyz_h_1(:,2) - cyl_center(2);
            cyl_center(1) = 0;
            cyl_center(2) = 0;
            centxyz_h_1 = new_xyz_h_stl;

            % set the pointCloud for the new shifted data & assign color
            ptCloud_humerus_stl = pointCloud(centxyz_h_1);
            points_h = zeros(ptCloud_humerus_stl.Count,3);
            ptCloud_humerus_stl.Color = uint8(ones(size(points_h)).*color_h);

            clear points_h xyzcent_h_1 XYZ_humerus new_xyz_h_stl
            clear trim_value ptCloud_humerus

            %% Temporary Sphere Data and Manipulation
            % Finding Z of Cylinder center to offset the sphere center
            F = scatteredInterpolant(centxyz_h_1(:,1),...
                                     centxyz_h_1(:,2),...
                                     centxyz_h_1(:,3),'nearest','none') ;
            % where (xi,yi) is the location for which you want z
            z_test_find = F(cyl_center(1),cyl_center(2));   
            % this will be the center of the hemisphere above the cylinder

            clear F

            %% Identifying Center of Sphere and Axes
            % creating the scenter of the sphere to make sure the top of the sphere is
            % tangent with the top of the humerus. 
            sph_center = [cyl_center(1) , cyl_center(2), z_test_find - fitted_radius]; 


            %% Plotting Sphere and Cylinder - Axis and Tangency Point
            % Create cylinder data and reset height of cylinder
            [x_cyl, y_cyl, z_cyl] = cylinder(fitted_radius); 
            z_cyl_new = z_cyl*sph_center(3);
            clear z_cyl

            % Make unit sphere for hemisphere plot
            [x,y,z] = sphere(40);
            % Scale to desired radius from the cylinderfit method 
            x = x * fitted_radius;
            y = y * fitted_radius;
            z = z * fitted_radius;
            % Segmenting out the bottom part of the sphere for the visual, only keeping
            % the top hemisphere
            z(z < 0) = NaN;
            x(any(isnan(z), 2), :) = [];
            y(any(isnan(z), 2), :) = [];
            z(any(isnan(z), 2), :) = [];
        end
        
        % figure to show the cylinder fitted to the humerus with different colors 
         figure('Name','"Bullet shape" fitted to the humerus point cloud','visible',visible)
            q = surf(x+sph_center(1),y+sph_center(2),z+sph_center(3));
            set(q, 'FaceAlpha', 0.3,'FaceColor', 'y')
            hold on 
            pcshow(ptCloud_humerus_fea);
            surf(x_cyl + cyl_center(1), y_cyl + cyl_center(2), z_cyl_new,...
                 'FaceAlpha', 0.3,'FaceColor', 'y')
            hold off
            axis equal;
            view(100,20)
            xlabel('X', 'FontSize', 16);
            ylabel('Y', 'FontSize', 16);
            zlabel('Z', 'FontSize', 16);
            title (titlename, 'FontSize', 20);

        filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig2-fitted-bullet-shape.png'];
        saveas(gcf,filename_png);
        clear filename_png
        if savefigure==1
            filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig2-fitted-bullet-shape.fig'];
            savefig(filename_matlab);
            clear filename_matlab
        end

        clear q mesh 
        
    else
        error('Limb not processed. Name must contain FEA, stl or mean.')
    end

    %% Humerus Bump Segmentation - Radius and Ulna Bumps
    % trimming data for upper region of humerus to segment out the bumps for
    % the radius and ulna. 
    [centxyz_h_1_bump] = trim((sph_center(3) - fitted_radius*bump_trim),...
                              '>', 3, centxyz_h_1);
                          
    % Given that all limbs are alignes in the same manner, we can directly 
    % assign the bump in "upper" part of y-axis as the radius.
    [radius_bdata_h_1] = trim((sph_center(2) + 1/3*fitted_radius), '>', 2, centxyz_h_1_bump);
    [ulna_bdata_h_1] = trim((sph_center(2) - 1/3*fitted_radius), '<', 2, centxyz_h_1_bump); 

    clear centxyz_h_1_bump

    %% Bump Segmentation
    % The respective ulna and radius bumps are then segmented out as whatever
    % is outside the sphere and the portion of the cylinder radius
    for q = 1:length(radius_bdata_h_1)
        if sqrt((radius_bdata_h_1(q,1) - sph_center(1))^2 + ...
                (radius_bdata_h_1(q,2) - sph_center(2))^2 + ...
                (radius_bdata_h_1(q,3) - sph_center(3))^2) > fitted_radius
           radius_bump_h_1(:,q) = radius_bdata_h_1((q),:); 
        end
    end

    %ulna
    for u = 1:length(ulna_bdata_h_1)
        if sqrt((ulna_bdata_h_1(u,1) - sph_center(1))^2 + ...
                (ulna_bdata_h_1(u,2) - sph_center(2))^2 + ...
                (ulna_bdata_h_1(u,3) - sph_center(3))^2) > fitted_radius 
           ulna_bump_h_1(:,u) = ulna_bdata_h_1((u),:); 
        end
    end

    clear q u radius_bdata_h_1 ulna_bdata_h_1

    %% Getting rid of Zeros
    % The segmentation of these bumps includes unnecessary zeros. 
    % Here we remove them
    colsWithZeros_top = any(radius_bump_h_1==0);
    radius_bump_h_2 = radius_bump_h_1(:, ~colsWithZeros_top)';

    colsWithZeros_bot = any(ulna_bump_h_1==0);
    ulna_bump_h_2 = ulna_bump_h_1(:, ~colsWithZeros_bot)';

    % Find the centroids of each bump
    top_bump_h_1_center = [mean(radius_bump_h_2(:,1)), ...
                           mean(radius_bump_h_2(:,2)), ...
                           mean(radius_bump_h_2(:,3))];
    bottom_bump_h_1_center = [mean(ulna_bump_h_2(:,1)), ...
                              mean(ulna_bump_h_2(:,2)), ...
                              mean(ulna_bump_h_2(:,3))];

    clear colsWithZeros_top colsWithZeros_bot
    clear radius_bump_h_1 ulna_bump_h_1 
    
    %% Check if bump assignment is correct
    % Compute point clouds without bumps
    centxyz_hum_empty_rad = setdiff(centxyz_h_1,radius_bump_h_2,'rows'); 
    centxyz_hum_empty_uln = setdiff(centxyz_h_1,ulna_bump_h_2,'rows'); 
    centxyz_hum_empty_both = setdiff(centxyz_hum_empty_uln,radius_bump_h_2,'rows'); 

    figure('Name','Checking correct radius bump assignment','visible',visible)
        % "bullet" shape
        surf(x+sph_center(1),y+sph_center(2),z+sph_center(3),...
             'FaceAlpha', 0.3,'FaceColor', 'y')
        hold on
        surf(x_cyl + cyl_center(1), y_cyl + cyl_center(2), z_cyl_new,...
             'FaceAlpha', 0.3,'FaceColor', 'y')
        % point clouds
        pcshow(radius_bump_h_2, color_r/255)
        if ~isnan(centxyz_u_1)
            pcshow(centxyz_r_1, color_r/255)
            pcshow(centxyz_u_1, [0.6 0.6 0.6])
        end
        pcshow(centxyz_hum_empty_rad, [0.6 0.6 0.6])
        % cylinder center and tangent point
        plot3(cyl_center(1), cyl_center(2), z_test_find,...
              'dy', 'MarkerSize', 5, 'LineWidth', 2);
        plot3(cyl_center(1), cyl_center(2),  cyl_center(3),...
              'dy', 'MarkerSize', 5, 'LineWidth', 2);
        % centroids of radius and ulna
        if ~isnan(centxyz_u_1)
            plot3(centroid_u_1(1),centroid_u_1(2),centroid_u_1(3),...
                  'om', 'MarkerSize', 10, 'LineWidth', 2)
            plot3(centroid_r_1(1),centroid_r_1(2),centroid_r_1(3),...
                  'om', 'MarkerSize', 10, 'LineWidth', 2)
        end
        hold off  
        axis equal;
        xlabel('X', 'FontSize', 16);
        ylabel('Y', 'FontSize', 16);
        zlabel('Z', 'FontSize', 16);
        title (titlename, 'FontSize', 20);

    filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig3-radius-bump-assignment.png'];
    saveas(gcf,filename_png);
    clear filename_png
    if savefigure==1
        filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig3-radius-bump-assignment.fig'];
        savefig(filename_matlab);
        clear filename_matlab
    end

    clear centroid_u_1 centroid_r_1 
    clear centxyz_hum_empty_rad

    %% Metrics calculations
    % Volume calculation of each bump
    [~, vol_rad] = convhull(radius_bump_h_2); 
    [~, vol_uln] = convhull(ulna_bump_h_2); 

    % Angle of bumps
    % Calculating lines between intersection points and sphere center
    top_bump_line = top_bump_h_1_center - sph_center;
    bottom_bump_line = bottom_bump_h_1_center - sph_center;
    % Calculating angles
    angle_bumps = atan2(norm(cross(top_bump_line, bottom_bump_line)), ...
                        dot(top_bump_line, bottom_bump_line));

    % Ulna height calculations
    % finding max point of Ulna bump
    ulna_max_idx = find(ulna_bump_h_2(:,3) == max(ulna_bump_h_2(:,3)));
    ulna_x = ulna_bump_h_2(:,1);
    ulna_y = ulna_bump_h_2(:,2);
    ulna_z = ulna_bump_h_2(:,3);
    ulna_max_point = [ulna_x(ulna_max_idx), ulna_y(ulna_max_idx), ulna_z(ulna_max_idx)];

    % finding height of max point from sphere center, both raw and normalized
    z_dist_ulna = (ulna_max_point(3) - sph_center(3));

    % Euclidian distance from sphere center to max ulna point, both raw and
    % normalized
    euc_dist_ulna = sqrt((ulna_max_point(1) - sph_center(1))^2 +...
                         (ulna_max_point(2) - sph_center(2))^2 +...
                         (ulna_max_point(3) - sph_center(3))^2); 

    % Radius height calculations
    % finding max point of radius bump
    radius_max_idx = find(radius_bump_h_2(:,3) == max(radius_bump_h_2(:,3)));
    radius_x = radius_bump_h_2(:,1);
    radius_y = radius_bump_h_2(:,2);
    radius_z = radius_bump_h_2(:,3);
    radius_max_point = [radius_x(radius_max_idx), radius_y(radius_max_idx), radius_z(radius_max_idx)];

    % finding height of max point from sphere center, both raw and normalized
    z_dist_radius = (radius_max_point(3) - sph_center(3));

    % Euclidian distance from sphere center to max radius point, both raw and
    % normalized
    euc_dist_radius = sqrt((radius_max_point(1) - sph_center(1))^2 +...
                           (radius_max_point(2) - sph_center(2))^2 +...
                           (radius_max_point(3) - sph_center(3))^2); 

    clear top_bump_h_1_center bottom_bump_h_1_center 
    clear top_bump_line bottom_bump_line
    clear ulna_max_idx ulna_x ulna_y ulna_z
    clear radius_max_idx radius_x radius_y radius_z

    %% Normalizing and saving metrics

    % Define values for normalization
    length_norm = fitted_radius;
    volume_norm = 2/3*pi*fitted_radius^3;

    % Compute variables to be saved and save in a substrcture
    newName = strcat(varname);
    metrics.(newName) = struct('UlnaVol',vol_uln, ...
                     'UlnaVolNorm', vol_uln/volume_norm, ...
                     'RadiusVol', vol_rad, ...
                     'RadiusVolNorm', vol_rad/volume_norm, ...
                     'AngleBumpsDeg', angle_bumps*180/pi, ...
                     'CylFitRadius', fitted_radius, ...
                     'EuclDistUlna', euc_dist_ulna, ...
                     'EuclDistUlnaNorm', euc_dist_ulna/length_norm, ...
                     'MaxVertDistUlna', z_dist_ulna, ...
                     'MaxVertDistUlnaNorm', z_dist_ulna/length_norm, ...
                     'EuclDistRadius', euc_dist_radius, ...
                     'EuclDistRadiusNorm', euc_dist_radius/length_norm, ...
                     'MaxVertDistRadius', z_dist_radius, ...
                     'MaxVertDistRadiusNorm', z_dist_radius/length_norm);

    clear newName vol_rad vol_uln angle_bumps length_norm volume_norm
    clear euc_dist_ulna z_dist_ulna euc_dist_radius z_dist_radius

    %% Point clouds of identified epicondyles

    % Compute centroids
    centroid_radius = mean(radius_bump_h_2); 
    centroid_ulna = mean(ulna_bump_h_2); 

    figure('Name','Point cloud of identified epicondyles','visible',visible);
        pcshow(radius_bump_h_2, color_r/255)
        hold on 
        pcshow(ulna_bump_h_2, color_u/255)
        xlabel('X', 'FontSize', 16);
        ylabel('Y', 'FontSize', 16);
        zlabel('Z', 'FontSize', 16);
        axis equal
        plot3(ulna_max_point(1), ulna_max_point(2), ulna_max_point(3),...
              'xy', 'MarkerSize', 10,  'LineWidth', 2)
        plot3(radius_max_point(1), radius_max_point(2), radius_max_point(3),...
             'xy', 'MarkerSize', 10,  'LineWidth', 2)
        plot3(centroid_radius(1),centroid_radius(2),centroid_radius(3),...
              '*g', 'MarkerSize', 10,  'LineWidth', 2)
        plot3(centroid_ulna(1),centroid_ulna(2),centroid_ulna(3),...
              '*g', 'MarkerSize', 10,  'LineWidth', 2)
        plot3(sph_center(1),sph_center(2),sph_center(3),...
              'om', 'MarkerSize', 10,  'LineWidth', 2)
        hold off
        title (titlename, 'FontSize', 20);
        view(-90, 0)

    filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig4-identified-bumps-isolated.png'];
    saveas(gcf,filename_png);
    clear filename_png
    if savefigure==1
        filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig4-identified-bumps-isolated.fig'];
        savefig(filename_matlab);
        clear filename_matlab
    end

    figure('Name','Point cloud of humerus with identified bumps','visible',visible);
        pcshow(centxyz_hum_empty_both, [0.6 0.6 0.6])
        hold on 
        pcshow(radius_bump_h_2, color_r/255)
        pcshow(ulna_bump_h_2, color_u/255)
        plot3(ulna_max_point(1), ulna_max_point(2), ulna_max_point(3),...
              'xy', 'MarkerSize', 10,  'LineWidth', 2)
        plot3(radius_max_point(1), radius_max_point(2), radius_max_point(3),...
             'xy', 'MarkerSize', 10,  'LineWidth', 2)
        plot3(centroid_radius(1),centroid_radius(2),centroid_radius(3),...
              '*g', 'MarkerSize', 10,  'LineWidth', 2)
        plot3(centroid_ulna(1),centroid_ulna(2),centroid_ulna(3),...
              '*g', 'MarkerSize', 10,  'LineWidth', 2)
        plot3(sph_center(1), sph_center(2), sph_center(3),... 
              'om','MarkerSize', 10,  'LineWidth', 2)
        hold off
        view(100,20)
        xlabel('X', 'FontSize', 16);
        ylabel('Y', 'FontSize', 16);
        zlabel('Z', 'FontSize', 16);
        title (titlename, 'FontSize', 20);
        axis equal

    filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig5-identified-bumps-humerus.png'];
    saveas(gcf,filename_png);
    clear filename_png
    if savefigure==1
        filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig5-identified-bumps-humerus.fig'];
        savefig(filename_matlab);
        clear filename_matlab
    end

    figure('Name','Point cloud of elbow with identified bumps','visible',visible);
        pcshow(centxyz_hum_empty_both, [0.6 0.6 0.6])
        hold on 
        pcshow(radius_bump_h_2, color_r/255)
        pcshow(ulna_bump_h_2,  color_u/255)
        if ~isnan(centxyz_u_1)
            pcshow(centxyz_r_1, color_r/255)
            pcshow(centxyz_u_1,  color_u/255)
        end
        hold off
        view(-50, 20)
        xlabel('X', 'FontSize', 16);
        ylabel('Y', 'FontSize', 16);
        zlabel('Z', 'FontSize', 16);
        title (titlename, 'FontSize', 20);
        axis equal

    filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig6-identified-bumps-limb.png'];
    saveas(gcf,filename_png);
    clear filename_png
    if savefigure==1
        filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig6-identified-bumps-limb.fig'];
        savefig(filename_matlab);
        clear filename_matlab
    end

    clear ulna_max_point radius_max_point pc_hum_empty
    clear centxyz_r_1 centxyz_u_1
    clear centxyz_hum_empty_rad centxyz_hum_empty_uln  centxyz_hum_empty_both
    clear radius_bump_h_2 ulna_bump_h_2

    %% Trim Data for sphere and cylinder
    % Trim data for cylinder analysis
    [centxyz_h_1_cyl] = trim(sph_center(3), '<', 3, centxyz_h_1);

    % Trim data for sphere analysis
    [centxyz_h_1_sph] = trim(sph_center(3), '>', 3, centxyz_h_1);

    clear centxyz_h_1
    % Keeping this data separate makes the normalization process simpler for
    % crafting the heatmaps shown below. 

    %% Cylindrical Normalization
    % Create the cylinder heatmap
    % Convert Cartesian coord. of stl surface to cylindrical coord.
    % Cylinder axis must be at origin (x=0,y=0) for this method to work
    [theta,rho_cyl] = cart2pol(centxyz_h_1_cyl(:,1) - cyl_center(1), ...
                               centxyz_h_1_cyl(:,2) - cyl_center(2));

    % Distance for color in the heatmap plot is radius of surface (rho_cyñ)  
    % minus radius of the fitted cylinder
    intensity_cyl = rho_cyl - fitted_radius;

    % Map points of surface onto cylinder by modifying all points to have same
    % radius (= radius of the cylinder)
    rho_cyl(:,1) = fitted_radius;

    % Convert back to Cartesian coord.(and shift back from origin)
    [x_cart,y_cart] = pol2cart(theta,rho_cyl);
    [x_cart_norm, y_cart_norm] = pol2cart(theta, 1);   %normalized heatmap has radius = 1
    clear rho_cyl theta

    % Normalized cylinder color maps
    centxyz_h_2_cyl_norm(:,1) = x_cart_norm; 
    centxyz_h_2_cyl_norm(:,2) = y_cart_norm;
    centxyz_h_2_cyl_norm(:,3) = centxyz_h_1_cyl(:,3)/fitted_radius;

    % Color maps set to radius of humerus
    centxyz_h_2_cyl = zeros(size(centxyz_h_1_cyl));
    centxyz_h_2_cyl(:,1) = x_cart + cyl_center(1);
    centxyz_h_2_cyl(:,2) = y_cart + cyl_center(2);
    centxyz_h_2_cyl(:,3) = centxyz_h_1_cyl(:,3);

    clear x_cart y_cart x_cart_norm y_cart_norm centxyz_h_1_cyl

    %% Sphere Normalization 
    % Creating the hemisphere heatmap 
    % Convert Cartesian coord. of stl surface to spherical coord.
    % Sphere center must be at origin (x=0,y=0, z=0) for this method to work
    [azimuth,elevation,rho_sph] = cart2sph(centxyz_h_1_sph(:,1) - sph_center(:,1), ...
                                           centxyz_h_1_sph(:,2) - sph_center(:,2), ...
                                           centxyz_h_1_sph(:,3) - sph_center(:,3));

    % Distance for color in the heatmap plot is radius of surface (rho_sph)  
    % minus radius of the fitted cylinder
    intensity_sph = rho_sph - fitted_radius;

    % Map points of surface onto cylinder by modifying all points to have same
    % radius (= radius of the cylinder)
    rho_sph(:,1) = fitted_radius;

    % Convert back to Cartesian coord.(and shift back from origin)
    [x_cart, y_cart, z_cart] = sph2cart(azimuth, elevation, rho_sph);
    [x_cart_norm, y_cart_norm(:,1), z_cart_norm(:,1)] = sph2cart(azimuth, elevation, 1);
    clear azimuth elevation rho_sph

    % Normalized cylinder color maps
    centxyz_h_2_sph_norm(:,1) = x_cart_norm; 
    centxyz_h_2_sph_norm(:,2) = y_cart_norm;
    centxyz_h_2_sph_norm(:,3) = z_cart_norm;

    % Color maps set to radius of humerus
    centxyz_h_2_sph(:,1) = x_cart + sph_center(1);
    centxyz_h_2_sph(:,2) = y_cart + sph_center(2);
    centxyz_h_2_sph(:,3) = z_cart + sph_center(3);

    clear x_cart y_cart z_cart x_cart_norm y_cart_norm z_cart_norm
    clear centxyz_h_1_sph

    %% Creating Cylinder and Spherical Heatmaps
    % These point clouds use the respective sphere and cylinder data with the
    % distance data to create heatmaps
    bullet_cyl = pointCloud(cat(3, centxyz_h_2_cyl(:,1),...
                                   centxyz_h_2_cyl(:,2),...
                                   centxyz_h_2_cyl(:,3)),...
                            'Intensity', intensity_cyl);
    bullet_sph = pointCloud(cat(3, centxyz_h_2_sph(:,1),...
                                   centxyz_h_2_sph(:,2),...
                                   centxyz_h_2_sph(:,3)),...
                            'Intensity', intensity_sph);

    clear centxyz_h_2_cyl centxyz_h_2_sph

    %% Creating Normalized Cylinder and Spherical Heatmaps
    % Create an array with coordinates and intensity value
    bullet_cyl_norm_array = horzcat(centxyz_h_2_cyl_norm,...
                                    intensity_cyl./fitted_radius);
    bullet_sph_norm_array = horzcat(centxyz_h_2_sph_norm,...
                                    intensity_sph./fitted_radius);

    % Trim array with cylinder coordinates to ensure correct aspect ratio 
    % of normalized bullet shape
    trim_height = (max(bullet_cyl_norm_array(:,3)) - norm_cylinder_height);
    bullet_cyl_norm_array_trim = trim(trim_height, '>', 3, bullet_cyl_norm_array);

    % Shift both cylinder and hemisphere to have z=0 at bottom of cylinder
    bullet_cyl_norm_array_trim(:,3) = bullet_cyl_norm_array_trim(:,3) ...
                                      - trim_height;
    bullet_sph_norm_array(:,3) = bullet_sph_norm_array(:,3) + ...
                                 sph_center(3)/fitted_radius - trim_height;

    % Convert to point clouds
    bullet_cyl_norm = pointCloud(cat(3, bullet_cyl_norm_array_trim(:,1),...
                                        bullet_cyl_norm_array_trim(:,2),...
                                        bullet_cyl_norm_array_trim(:,3)),...
                                 'Intensity', bullet_cyl_norm_array_trim(:,4));          
    bullet_sph_norm = pointCloud(cat(3, bullet_sph_norm_array(:,1),...
                                        bullet_sph_norm_array(:,2),...
                                        bullet_sph_norm_array(:,3)),...
                                  'Intensity', bullet_sph_norm_array(:,4));             


    clear centxyz_h_2_cyl_norm centxyz_h_2_sph_norm
    clear intensity_cyl intensity_sph
    clear bullet_cyl_norm_array bullet_sph_norm_array 
    clear bullet_cyl_norm_array_trim trim_height

    %% Heatmap on standarized "bullet" surface 
    figure('Name','Heatmap on "bullet" shape','visible',visible)
        pcshow(bullet_cyl)
        hold on 
        pcshow(bullet_sph)
        xlabel('X', 'FontSize', 16);
        ylabel('Y', 'FontSize', 16);
        zlabel('Z', 'FontSize', 16);
        axis equal;
        hold off
        t = colorbar;
        t.FontSize = 18;
        t.Color = legend_text_color;
        ylabel(t, 'Distance to reference surface (um)')
        title (titlename, 'FontSize', 20);
        view([-40, 20])

    filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig7-bullet-heatmap-points.png'];
    saveas(gcf,filename_png);
    clear filename_png
    if savefigure==1
        filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig7-bullet-heatmap-points.fig'];
        savefig(filename_matlab);
        clear filename_matlab
    end

    %% Heatmap on standarized and normalized "bullet" surface 
    % Add centroids of radius and ulna. Project them to bullet shape
    origin_sphere = zeros(1,3);

    %Cylinder axis has already been shifted to coincide with z axis
    %origin_sphere(1) = (bullet_sph.XLimits(1) + bullet_sph.XLimits(2)) / 2;
    %origin_sphere(2) = (bullet_sph.YLimits(1) + bullet_sph.YLimits(2)) / 2;
    origin_sphere(3) = bullet_sph.ZLimits(1);

    centroid_radius_aux = centroid_radius - origin_sphere;
    centroid_ulna_aux = centroid_ulna - origin_sphere;

    % Convert to spherical coordinates
    [azimuth_cr,elevation_cr,~] = cart2sph(centroid_radius_aux(1), ...
                                           centroid_radius_aux(2), ...
                                           centroid_radius_aux(3));
    [azimuth_cu,elevation_cu,~] = cart2sph(centroid_ulna_aux(1), ...
                                           centroid_ulna_aux(2), ...
                                           centroid_ulna_aux(3));

    % Set radius to 1 and convert back to Cartesian coord.
    [centroid_radius_aux(1),...
     centroid_radius_aux(2),...
     centroid_radius_aux(3)] = sph2cart(azimuth_cr,elevation_cr,1);

    [centroid_ulna_aux(1), ...
     centroid_ulna_aux(2), ...
     centroid_ulna_aux(3)] = sph2cart(azimuth_cu,elevation_cu,1);

    % Add height such that it coincides with hemisphere of normalized bullet 
    centroid_radius_norm = centroid_radius_aux;
    centroid_ulna_norm = centroid_ulna_aux;
    centroid_radius_norm(3) = centroid_radius_aux(3) +  bullet_sph_norm.ZLimits(1);
    centroid_ulna_norm(3) = centroid_ulna_aux(3) + bullet_sph_norm.ZLimits(1);

    clear origin_sphere
    clear azimuth_cr elevation_cr radius_cr centroid_radius_aux
    clear azimuth_cu elevation_cu radius_cu centroid_ulna_aux

    % Normalized heatmap and original humerus surface
    figure('Name','Heatmap on normalized "bullet" shape','visible',visible);
        pcshow(bullet_cyl_norm)
        hold on 
        pcshow(bullet_sph_norm)
        p = plot3(centroid_radius_norm(1), ...
                  centroid_radius_norm(2), ...
                  centroid_radius_norm(3),'ok');
        p.LineWidth = 5;
        p.MarkerSize = 18;
        p = plot3(centroid_ulna_norm(1), ...
                  centroid_ulna_norm(2), ...
                  centroid_ulna_norm(3),'+k');
        p.LineWidth = 5;
        p.MarkerSize = 18;
        hold off
        xlabel('X', 'FontSize', 16)
        ylabel('Y', 'FontSize', 16)
        zlabel('Z', 'FontSize', 16)
        axis equal
        xlim([-1 1])
        ylim([-1 1])
        zlim([0 inf]) 
        title ({titlename;'+ = ulna, o = radius'}, 'FontSize', 20);

        t = colorbar;
        t.FontSize = 18;
        t.Color = 'white';
        ylabel(t, 'Normalized distance to reference surface')

        ax = gca(); 
        ax.CLim = [-colorbar_limit colorbar_limit];
        ax.FontSize = 18; 

    filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig8-bullet-heatmap-points-normalized.png'];
    saveas(gcf,filename_png);
    clear filename_png
    if savefigure==1
        filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig8-bullet-heatmap-points-normalized.fig'];
        savefig(filename_matlab);
        clear filename_matlab
    end

    clear t p ax

    %% Flattened Heatmap
    % Save Cartesian Coord of bullet point cloud and intensity values into arrays
    bullet_cyl_array_norm = zeros(bullet_cyl_norm.Count,4);
    bullet_sph_array_norm = zeros(bullet_sph_norm.Count,4);

    for i=1:3
        bullet_cyl_array_norm(:,i) = bullet_cyl_norm.Location(:,:,i);
        bullet_sph_array_norm(:,i) = bullet_sph_norm.Location(:,:,i);
    end
    clear i
    bullet_cyl_array_norm(:,4) = bullet_cyl_norm.Intensity(:,1);
    bullet_sph_array_norm(:,4) = bullet_sph_norm.Intensity(:,1);


    % "Peel" open cylinder: convert to polar coord
    [theta,~] = cart2pol(bullet_cyl_array_norm(:,1), ...
                             bullet_cyl_array_norm(:,2));

    flat_cyl_array_norm = zeros(bullet_cyl_norm.Count,3);
    flat_cyl_array_norm(:,1) = theta;
    flat_cyl_array_norm(:,2) = bullet_cyl_array_norm(:,3);
    flat_cyl_array_norm(:,3) = bullet_cyl_array_norm(:,4);

    clear theta bullet_cyl_array_norm

    % Translate hemisphere to origin of coord (0,0,0)
    origin_sphere_norm = bullet_cyl_norm.ZLimits(2);
    bullet_sph_array_norm(:,3) = bullet_sph_array_norm(:,3) - origin_sphere_norm;

    % Project hemisphere to flat surface: convert to spherical coord
    [azimuth,elevation_sph,~] = cart2sph(bullet_sph_array_norm(:,1), ...
                                             bullet_sph_array_norm(:,2), ...
                                             bullet_sph_array_norm(:,3));

    flat_sph_array_norm = zeros(bullet_sph_norm.Count,3);
    flat_sph_array_norm(:,1) = azimuth;
    flat_sph_array_norm(:,2) = elevation_sph; 
    flat_sph_array_norm(:,3) = bullet_sph_array_norm(:,4);

    % Translate back to original vertical position of sphere.
    flat_sph_array_norm(:,2) = flat_sph_array_norm(:,2) + origin_sphere_norm;

    % Add the two flattened parts together
    flat_array_norm = cat(1,flat_cyl_array_norm,flat_sph_array_norm);

    % Add "fake" data point (= 0) in bottom left corner to ensure 
    % continuous heatmaps will all be the same, even when limb has shorter
    % normalized bullet
    % IMPORTANT: this affects our mean computation, but we will not be 
    % analyzing the corner bin data of our heatmap
    flat_array_norm (end+1,1) = -pi;
    flat_array_norm (end+1,2) = 0;
    flat_array_norm (end+1,3) = 0;

    % Add centroids of epicondyles unto flattened out map
    % Translate to origin of coord (0,0,0)
    centroid_radius_norm(3) = centroid_radius_norm(3)-origin_sphere_norm;
    centroid_ulna_norm(3) = centroid_ulna_norm(3)-origin_sphere_norm;

    % Project points on hemisphere to flat surface: convert to spherical coord
    [azimuth_rc,elevation_rc,~] = cart2sph(centroid_radius_norm(1), ...
                                           centroid_radius_norm(2), ...
                                           centroid_radius_norm(3));
    [azimuth_uc,elevation_uc,~] = cart2sph(centroid_ulna_norm(1), ...
                                           centroid_ulna_norm(2), ...
                                           centroid_ulna_norm(3));

    flat_centroid_radius_norm = zeros(1,2);
    flat_centroid_ulna_norm = zeros(1,2);

    flat_centroid_radius_norm(1) = azimuth_rc;
    flat_centroid_radius_norm(2) = elevation_rc;

    flat_centroid_ulna_norm(1) = azimuth_uc;
    flat_centroid_ulna_norm(2) = elevation_uc;

    % Translate back to original vertical position of sphere.
    flat_centroid_radius_norm(2) = flat_centroid_radius_norm(2) + origin_sphere_norm;
    flat_centroid_ulna_norm(2) = flat_centroid_ulna_norm(2) + origin_sphere_norm;
    
    clear azimuth elevation_sph bullet_sph_array_norm
    clear flat_cyl_array_norm flat_sph_array_norm          
    clear azimuth_rc elevation_rc centroid_radius_norm
    clear azimuth_uc elevation_uc centroid_ulna_norm origin_sphere_norm

    % figure call 
    figure('Name','Flattened out heatmap','visible',visible);
        scatter(flat_array_norm(:,1),...
                flat_array_norm(:,2),...
                1, flat_array_norm(:,3), 'filled')
        hold on
        p = plot(flat_centroid_radius_norm(1), ...
                 flat_centroid_radius_norm(2),'ok');
        p.LineWidth = 1.5;
        p = plot(flat_centroid_ulna_norm(1), ...
                 flat_centroid_ulna_norm(2),'+k');
        p.LineWidth = 1.5;
        hold off
        axis equal
        xlabel('reference surface perimeter', 'FontSize', 16)
        ylabel('proximal-distal direction', 'FontSize', 16)
        title ({titlename;'+ = ulna, o = radius'}, 'FontSize', 20);
        colormap(jet)

        t = colorbar;
        t.FontSize = 18;
        ylabel(t, 'Normalized distance to reference surface')
        ax = gca(); 
        ax.CLim = [-colorbar_limit colorbar_limit];
        ax.FontSize = 18; 

    filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig9-heatmap-points.png'];
    saveas(gcf,filename_png);
    clear filename_png
    if savefigure==1
        filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig9-heatmap-points.fig'];
        savefig(filename_matlab);
        clear filename_matlab
    end

    clear ax t p

    %% Convert flattened out heatmap to continuous plot

    figure('Name','Binned scatter plot showing data point count','visible',visible);
        binscatter(flat_array_norm(:,1), flat_array_norm(:,2),NumBins)
        hold on
        p = plot(flat_centroid_radius_norm(1), ...
                 flat_centroid_radius_norm(2),'ok');
        p.LineWidth = 1.5;
        p = plot(flat_centroid_ulna_norm(1), ...
                 flat_centroid_ulna_norm(2),'+k');
        p.LineWidth = 1.5;
        axis equal
        xlabel('reference surface perimeter', 'FontSize', 16)
        ylabel('proximal-distal direction', 'FontSize', 16)

        title_txt = strcat('+ = ulna, o = radius; bin width: ', num2str(bin_width));
        title ({titlename;title_txt}, 'FontSize', 20);
        t = colorbar;
        t.FontSize = 16;
        ylabel(t, 'bin counts')

        ax = gca(); 
        ax.FontSize = 16; 

    filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig10-heatmap-point-count.png'];
    saveas(gcf,filename_png);
    clear filename_png
    if savefigure==1
        filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig10-heatmap-point-count.fig'];
        savefig(filename_matlab);
        clear filename_matlab
    end

    clear N title_txt t ax p

    % Order data in array for x-column
    %flat_array_norm_sorted = sortrows(flat_array_norm);
    [N_x,edges_x] = histcounts(flat_array_norm(:,1),NumBins(1));
    [N_y,edges_y] = histcounts(flat_array_norm(:,2),NumBins(2));

    % Bin data according to x-column & then to y-column
    binned_data = cell(NumBins(1),NumBins(2));    
    binned_data_mean = zeros(NumBins(1), NumBins(2));
    %binned_data_median = zeros(NumBins(1), NumBins(2));

    for i = 1:(size(N_x,2))
        % Bin data according to x-column
        subarray = flat_array_norm( (flat_array_norm(:,1) >= edges_x(1,i) ) &...
                                    (flat_array_norm(:,1) <  edges_x(1,i+1) ), :);

        for j = 1:(size(N_y,2))
            % Bin data according to y-column
            binned_data{i,j} = subarray (...
                           (subarray (:,2) >= edges_y(1,j) ) &...
                           (subarray (:,2) <  edges_y(1,j+1) ), :);    

            aux = binned_data{i,j};
            binned_data_mean(i,j) = mean(aux(:,3));
            %binned_data_median(i,j) = median(aux(:,3));
        end
    end
    clear i j N_y N_x aux subarray edges_x edges_y
    
    %% Plot mean values heatmap
    % heatmaps can't be combined with graphic elements like points, so 
    % we can't add the projected centroids of the radius and ulna condyles
    title_txt = strcat('Mean normalized distance to reference surface, bin width: ', num2str(bin_width));

    figure('Name','Continuous flattened out heatmap (mean)','visible',visible);
        hm = heatmap(flip(binned_data_mean'));
        colormap(jet)
        hm.ColorLimits = [-colorbar_limit colorbar_limit];
        hm.XLabel ='reference surface perimeter';
        hm.YLabel ='proximal-distal direction';
        hm.Title ={titlename;title_txt};

        ax = gca(); 
        ax.FontSize = 16; 
        % Remove tick labels from x and y axis
        ax.XDisplayLabels = nan(size(ax.XDisplayData));
        ax.YDisplayLabels = nan(size(ax.YDisplayData));
        
    filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig11-heatmap-binned-means.png'];
    saveas(gcf,filename_png);
    clear filename_png
    if savefigure==1
        filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig11-heatmap-binned-means.fig'];
        savefig(filename_matlab);
        clear filename_matlab
    end

    clear fig5 ax hm title_txt 
    
%% Plot mean values heatmap
    
    % Fill missing
    HeatmapMeanFill = fillmissing(binned_data_mean,'linear');
    
    
    title_txt = strcat('Mean normalized distance to reference surface, bin width: ', num2str(bin_width));

    figure('Name','Continuous flattened out heatmap (mean,filled)','visible',visible);
        hm = heatmap(flip(HeatmapMeanFill'));
        colormap(jet)
        hm.ColorLimits = [-colorbar_limit colorbar_limit];
        hm.XLabel ='reference surface perimeter';
        hm.YLabel ='proximal-distal direction';
        hm.Title ={titlename;title_txt};

        ax = gca(); 
        ax.FontSize = 16; 
        % Remove tick labels from x and y axis
        ax.XDisplayLabels = nan(size(ax.XDisplayData));
        ax.YDisplayLabels = nan(size(ax.YDisplayData));
        
    filename_png=[pwd '/' filepath '/' titlename '_Part1_Fig12-heatmap-binned-means_filled.png'];
    saveas(gcf,filename_png);
    clear filename_png
    if savefigure==1
        filename_matlab=[pwd '/' filepath '/' titlename '_Part1_Fig12-heatmap-binned-means-filled.fig'];
        savefig(filename_matlab);
        clear filename_matlab
    end

    clear ax hm title_txt 

    
    %% Change variable names and save relevant heatmap data to struct 

    % Heatmaps
    newName1 = strcat(varname,'_HeatmapData');
    newName2 = strcat(varname,'_HeatmapMean');
    newName3 = strcat(varname,'_CentroidRadius');
    newName4 = strcat(varname,'_CentroidUlna');
    newName5 = strcat(varname,'_BinWidth');
    newName6 = strcat(varname,'_CylHeight');
    heatmaps.(newName1)=binned_data;
    heatmaps.(newName2)=binned_data_mean;
    heatmaps.(newName3)=flat_centroid_radius_norm;
    heatmaps.(newName4)=flat_centroid_ulna_norm;
    heatmaps.(newName5)=bin_width;
    heatmaps.(newName6)=norm_cylinder_height;
    
    clear newName1 newName2 newName3 newName4 newName5 newName6
    clear binned_data binned_data_mean 
    clear flat_centroid_radius_norm flat_centroid_ulna_norm

    
    clear bullet_cyl bullet_sph bullet_cyl_norm bullet_sph_norm 
    clear centroid_radius centroid_ulna flat_array_norm
    
    % Save data in struct variables to .mat file
    % We do it inside the loop so that if an error occurs, previous 
    % limb results aren't lost.
    cd results/
    save(filename,'heatmaps','metrics','-append');
    cd ..
    
    % Print info to Command Window
    X = sprintf('Analysis of %s complete.',titlename);
    disp(X)
    clear X
    
end

% Print info to Command Window
disp('Full analysis complete.');


%% Clear variables
clear all;
