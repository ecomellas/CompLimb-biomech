%% Header
% Plotting growth heatmaps of predicted humerus surfaces
%
% Ester Comellas - 22/05/2021, Northeastern University
% e.comellas@northeastern.edu

close all;

%% Load and plot bullet heatmaps
load ('MeanSurfacesDMSOandGSK.mat'); 
visible = 'on';
fea_mesh_file = { 'FEA_cav_coarse.vtk'; ...   % cavitation
                  'FEA_grown__small-dt__km1e-5__kb2e-5.vtk';... %DMSO
                  'FEA_grown__small-dt__km0__kb2e-5.vtk'};       %GSK

factor_mesh = [ 0; ...   % cavitation
                500;... % DMSO
                500];   % GSK

savefigure = 0;
legend_text_color = 'white';
colorbar_limit = 0.65;

%Define num. of refinements to upsample the point clouds of the FEA results
max_refine = 10;
FEA_radius_DMSO = 250;
FEA_radius_GSK = 250;

% Define flat array size for continuous heatmaps based on user data
norm_cylinder_height = 4;
bin_width = 0.03;
max_horiz = 2*pi; % Perimeter of the normalized bullet shape (radius=1)
max_vert = norm_cylinder_height + pi/2; % Height of cylinder + 1/2 perimeter of hemisphere
NumBins = [floor(max_horiz/bin_width), floor(max_vert/bin_width)];
clear max_horiz max_vert

%% Create results folder if it doesn't exist
if (exist('results', 'dir')~=7) 
    mkdir results;
end
filepath = 'results/';

%% Load FEA meshes
fea_mesh = cell(3,1);
fea_growth = cell(3,1);

for iter=1:3

    % Read in cav vtk file
    fid = fopen(fea_mesh_file{iter},'r');

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

    % counter k not reset because CELLS always comes after POINTS
    found = 1;
    while found
        if isempty(elem_pos{k,1}) == 0
            idx_elem = k;
            found = 0;
        end
        k = k + 1;
    end

    % counter k not reset because CELL_TYPE always comes after POINTS
    found = 1;
    while found
        if isempty(elem_type_pos{k,1}) == 0
            idx_elem_type = k;
            found = 0;
        end
        k = k + 1;
    end

    % counter k not reset because displacement always comes after
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
    mesh.vertices = input_mesh_nodes + factor_mesh(iter)*input_mesh_disp;
    mesh.faces = input_mesh_elems;
    
    % Create array for coordinates and growth measure
    coord_growth = zeros(size(input_mesh_nodes,1),4);
    coord_growth(:,1:3) = input_mesh_nodes*rotz(180);  % Rotate to match the experimental alignment
    coord_growth(:,4) = sqrt( input_mesh_disp(:,1).^2 + ...
                              input_mesh_disp(:,2).^2 + ...
                              input_mesh_disp(:,3).^2 );
                          
    % Upsample cloud point
    max_faces = size(mesh.faces,1);
    coord_per_elem = (4+max_refine)*max_refine;
    new_coord = NaN(max_faces*coord_per_elem,4);
    tot_count = 0;
    for e = 1:max_faces

        elem_nodes = mesh.faces(e,:);
        vertex = NaN(size(elem_nodes,2),4);
        for v = 1:size(elem_nodes,2)
            vertex(v,:) = coord_growth(elem_nodes(v),:);
        end
        clear v elem_nodes
        new_coord_elem = NaN(coord_per_elem,4);

        edge_1 = vertex(2,:)-vertex(1,:);
        edge_2 = vertex(3,:)-vertex(4,:);
        coord_edge_1 = NaN(max_refine+2,4);
        coord_edge_1(1,:) = vertex(1,:);
        coord_edge_1(end,:) = vertex(2,:);

        coord_edge_2 = NaN(max_refine+2,4);
        coord_edge_2(1,:) = vertex(4,:);
        coord_edge_2(end,:) = vertex(3,:);
        clear vertex

        for n = 1:max_refine
            coord_edge_1(1+n,:) = coord_edge_1(n,:) + ...
                                  edge_1/((max_refine+1)); 
            coord_edge_2(1+n,:) = coord_edge_2(n,:) + ...
                                  edge_2/((max_refine+1));
        end
        new_coord_elem(1:max_refine,:) = coord_edge_1(2:1+max_refine,:);
        new_coord_elem(max_refine+1:2*max_refine,:) = coord_edge_2(2:1+max_refine,:);

        clear edge_1 edge_2 n

        for k = 1:(max_refine+2)
            edge_int = coord_edge_2(k,:) - coord_edge_1(k,:);
            coord_int = NaN(max_refine+2,4);
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
    total_coord_growth = cat(1,coord_growth,new_coord_no_dupl); 
    
    % Move mesh to align with humeus surface
    mesh_new = mesh;
    FEA_xyz = mesh_new.vertices*rotz(160);
    z_shift = min(FEA_xyz(:,3));
    FEA_xyz_new = FEA_xyz;
    FEA_xyz_new(:,3) = FEA_xyz(:,3) - z_shift;
    mesh_new.vertices = FEA_xyz_new;

    fea_mesh{iter} = mesh_new;
    fea_growth{iter} = total_coord_growth;
    
    clear mesh mesh_new z_shift FEA_xyz FEA_xyz_new z_shift 
    clear coord_growth total_coord_growth new_coord_no_dupl
    clear input_mesh_nodes input_mesh_disp input_mesh_elems
end
clear iter

%% Plot DMSO & GSK growth measures on original mesh


% Translate z coordinates and remove bottom points for DMSO
XYZ_coord = fea_growth{2}(:,1:3);  

% Normalize magnitude of distance to maximum value in a given humerus 
% for intensity value of point cloud 
XYZ_inten = fea_growth{2}(:,4)/max(fea_growth{2}(:,4));  %DMSO normalized with max DMSO

% Create array with translated coord
XYZ_trans = zeros(size(XYZ_coord,1),4);
XYZ_trans(:,1:2) = XYZ_coord(:,1:2);
XYZ_trans(:,3) = XYZ_coord(:,3) - min(XYZ_coord(:,3));
XYZ_trans(:,4) = XYZ_inten;

% Remove bottom 2% of humerus
trim_value = (max(XYZ_trans(:,3))-min(XYZ_trans(:,3)))*0.02;
XYZ_trans(XYZ_trans(:,3) < trim_value,:) = [];

% Create point Cloud
DMSO_growth_ptCloud = pointCloud(XYZ_trans(:,1:3),'Intensity', XYZ_trans(:,4));
% Sphere center was at 0,0,0 and has now shifted vertically
DMSO_sph_center = [0,0,-min(XYZ_coord(:,3))];
clear XYZ_trans XYZ_coord XYZ_inten trim_value


% Repeat for GSK (in case mesh is not the same)
XYZ_coord = fea_growth{3}(:,1:3);  
%XYZ_inten = fea_growth{3}(:,4)/max(fea_growth{3}(:,4));   %GSK normalized with max GSK
XYZ_inten = fea_growth{3}(:,4)/max(fea_growth{2}(:,4));   %GSK normalized with max DMSO
XYZ_trans = zeros(size(XYZ_coord,1),4);
XYZ_trans(:,1:2) = XYZ_coord(:,1:2);
XYZ_trans(:,3) = XYZ_coord(:,3) - min(XYZ_coord(:,3));
XYZ_trans(:,4) = XYZ_inten;
trim_value = (max(XYZ_trans(:,3))-min(XYZ_trans(:,3)))*0.02;
XYZ_trans(XYZ_trans(:,3) < trim_value,:) = [];
GSK_growth_ptCloud = pointCloud(XYZ_trans(:,1:3),'Intensity', XYZ_trans(:,4));
GSK_sph_center = [0,0,-min(XYZ_coord(:,3))];
clear XYZ_trans XYZ_coord XYZ_inten trim_value


% Plot DMSO                           
titlename = 'DMSO growth map';
figure('Name','DMSO humerus surface growth','visible',visible)
            pcshow(DMSO_growth_ptCloud)
            xlabel('X', 'FontSize', 16);
            ylabel('Y', 'FontSize', 16);
            zlabel('Z', 'FontSize', 16);
            axis equal;
            t = colorbar;
            t.FontSize = 18;
            t.Color = legend_text_color;
            ylabel(t, 'Magnitude of displacement (normalized to max)')
            colormap(parula)
            
            ax = gca(); 
            ax.CLim = [0 1];
            ax.FontSize = 18; 
            
            title (titlename, 'FontSize', 20);
            view([-40, 20])
 
 
filename_png=[pwd '/' filepath '/Fig100-DMSO-growth.png'];
saveas(gcf,filename_png);

if savefigure==1
    filename_matlab=[pwd '/' filepath '/Fig100-DMSO-growth.fig'];
    savefig(filename_matlab);
end                      

clear DMSO_growth t titlename filename_png
          
% Plot GSK
titlename = 'GSK growth map';
figure('Name','GSK humerus surface growth','visible',visible)
            pcshow(GSK_growth_ptCloud )
            xlabel('X', 'FontSize', 16);
            ylabel('Y', 'FontSize', 16);
            zlabel('Z', 'FontSize', 16);
            axis equal;
            t = colorbar;
            t.FontSize = 18;
            t.Color = legend_text_color;
            ylabel(t, 'Magnitude of displacement (normalized to max)')
            colormap(parula)%colormap(flipud(autumn))
                   
            ax = gca(); 
            ax.CLim = [0 1];
            ax.FontSize = 18; 

        
            title (titlename, 'FontSize', 20);
            view([-40, 20])
            
filename_png=[pwd '/' filepath '/Fig100-GSK-growth.png'];
saveas(gcf,filename_png);

if savefigure==1
    filename_matlab=[pwd '/' filepath '/Fig100-GSK-growth.fig'];
    savefig(filename_matlab);
end              

clear GSK_growth t titlename filename_png

%% Flatten out growth heatmap 

for iter=1:2
    
    % Create "bullet shape"
    
    % Make cylinder
    switch (iter)
        case 1
            FEA_radius = FEA_radius_DMSO;
            sph_center = DMSO_sph_center;
            growth_ptCloud = DMSO_growth_ptCloud;
            name = 'DMSO';
        otherwise
            FEA_radius = FEA_radius_GSK;
            sph_center = GSK_sph_center;
            growth_ptCloud = GSK_growth_ptCloud;
            name = 'GSK';
            
    end
    [x_cyl, y_cyl, z_cyl] = cylinder(FEA_radius); 
    z_cyl = z_cyl * sph_center(3);
    cyl_center = [0,0];
    % Make unit sphere for hemisphere plot
    [x_sph,y_sph,z_sph] = sphere(40);
    % Scale to desired radius from the cylinderfit method 
    x_sph = x_sph * FEA_radius;
    y_sph = y_sph * FEA_radius;
    z_sph = z_sph * FEA_radius;
    % Segmenting out the bottom part of the sphere for the visual, only keeping
    % the top hemisphere
    z_sph(z_sph < 0) = NaN;
    x_sph(any(isnan(z_sph), 2), :) = [];
    y_sph(any(isnan(z_sph), 2), :) = [];
    z_sph(any(isnan(z_sph), 2), :) = [];


    % figure to show the cylinder fitted to the humerus with different colors 
    titlename =  strcat(name, ' fitted "bullet" shape');
    figure('Name','"Bullet shape" fitted to mesh','visible',visible)
    q = surf(x_sph + sph_center(1),...
     y_sph + sph_center(2),...
     z_sph + sph_center(3));
    set(q, 'FaceAlpha', 0.3,'FaceColor', 'w')
    hold on 
    surf(x_cyl + cyl_center(1), y_cyl + cyl_center(2), z_cyl,...
    'FaceAlpha', 0.3,'FaceColor', 'w')
    pcshow(growth_ptCloud);
    hold off
    xlabel('X', 'FontSize', 16);
    ylabel('Y', 'FontSize', 16);
    zlabel('Z', 'FontSize', 16);
    axis equal;
    t = colorbar;
    t.FontSize = 18;
    t.Color = legend_text_color;
    ylabel(t, 'Magnitude of displacement (normalized to max)')
    colormap(parula)%colormap(flipud(autumn))

    ax = gca(); 
    ax.CLim = [0 1];
    ax.FontSize = 18; 

    title (titlename, 'FontSize', 20);
    view([-40, 20])

    filename_png=[pwd '/' filepath '/Fig101-' name '-cylinder-fit.png'];
    saveas(gcf,filename_png);

    if savefigure==1
        filename_matlab=[pwd '/' filepath '/Fig101-' name '-cylinder-fit.fig'];
        savefig(filename_matlab);
    end              
    clear ax t filename_matlab filename_png titlename
    
    
    % Map growth measure onto bullet shape
    
    % Create the cylinder heatmap
    % Extract coord corresponding to cylindrical part
    XYZ_cyl = zeros(size(growth_ptCloud.Location,1),4);
    XYZ_cyl(:,1:3) = growth_ptCloud.Location;
    XYZ_cyl(:,4) = growth_ptCloud.Intensity;
    XYZ_cyl(XYZ_cyl(:,3) >= sph_center(3),:) = [];
    
    % Convert Cartesian coord. to cylindrical coord.
    % Cylinder axis must be at origin (x=0,y=0) for this method to work
    [theta,rho_cyl] = cart2pol(XYZ_cyl(:,1) - cyl_center(1), ...
                               XYZ_cyl(:,2) - cyl_center(2));

    % Distance for color in the original heatmap plot is radius of surface 
    % (rho_cyl) minus radius of the fitted cylinder
    intensity_cyl = rho_cyl - FEA_radius;

    % Map points of surface onto cylinder by modifying all points to have same
    % radius (= radius of the cylinder)
    rho_cyl(:,1) = FEA_radius;

    % Convert back to Cartesian coord.(and shift back from origin)
    [x_cart,y_cart] = pol2cart(theta,rho_cyl);
    [x_cart_norm, y_cart_norm] = pol2cart(theta, 1);   %normalized heatmap has radius = 1
    clear rho_cyl theta

    % Normalized cylinder color maps
    XYZ_cyl_norm(:,1) = x_cart_norm; 
    XYZ_cyl_norm(:,2) = y_cart_norm;
    XYZ_cyl_norm(:,3) = XYZ_cyl(:,3)/FEA_radius;

    % Color maps set to radius of humerus
    XYZ_cyl(:,1) = x_cart + cyl_center(1);
    XYZ_cyl(:,2) = y_cart + cyl_center(2);

    clear x_cart y_cart x_cart_norm y_cart_norm
    
    % Create the hemisphere heatmap 
    % Extract coord corresponding to cylindrical part
    XYZ_sph = zeros(size(growth_ptCloud.Location,1),4);
    XYZ_sph(:,1:3) = growth_ptCloud.Location;
    XYZ_sph(:,4) = growth_ptCloud.Intensity;
    XYZ_sph(XYZ_sph(:,3) < sph_center(3),:) = [];
    
    % Convert Cartesian coord. of stl surface to spherical coord.
    % Sphere center must be at origin (x=0,y=0, z=0) for this method to work
    [azimuth,elevation,rho_sph] = cart2sph(XYZ_sph(:,1) - sph_center(:,1), ...
                                           XYZ_sph(:,2) - sph_center(:,2), ...
                                           XYZ_sph(:,3) - sph_center(:,3));

    % Distance for color in the heatmap plot is radius of surface (rho_sph)  
    % minus radius of the fitted cylinder
    intensity_sph = rho_sph - FEA_radius;

    % Map points of surface onto cylinder by modifying all points to have same
    % radius (= radius of the cylinder)
    rho_sph(:,1) = FEA_radius;

    % Convert back to Cartesian coord.(and shift back from origin)
    [x_cart, y_cart, z_cart] = sph2cart(azimuth, elevation, rho_sph);
    [x_cart_norm, y_cart_norm(:,1), z_cart_norm(:,1)] = sph2cart(azimuth, elevation, 1);
    clear azimuth elevation rho_sph

    % Normalized cylinder color maps
    sph_center_norm = sph_center./FEA_radius;
    XYZ_sph_norm(:,1) = x_cart_norm + sph_center_norm(1); 
    XYZ_sph_norm(:,2) = y_cart_norm + sph_center_norm(2);
    XYZ_sph_norm(:,3) = z_cart_norm + sph_center_norm(3);

    % Color maps set to radius of humerus
    XYZ_sph(:,1) = x_cart + sph_center(1);
    XYZ_sph(:,2) = y_cart + sph_center(2);
    XYZ_sph(:,3) = z_cart + sph_center(3);

    clear x_cart y_cart z_cart x_cart_norm y_cart_norm z_cart_norm
    
    % Join cylinder and spherical coordinates into a single array
    XYZ = cat(1,XYZ_sph,XYZ_cyl);
    XYZ_norm = cat(1,XYZ_sph_norm,XYZ_cyl_norm);
    bullet_surf = cat(1,intensity_sph,intensity_cyl);
    bullet_surf_norm = bullet_surf./FEA_radius;
    % Generate variables needed later
    intensity_cyl_norm = intensity_cyl./FEA_radius;
    intensity_sph_norm = intensity_sph./FEA_radius;
    
    original_bullet = pointCloud(XYZ(:,1:3), 'Intensity', bullet_surf);
    growth_bullet = pointCloud(XYZ(:,1:3), 'Intensity', XYZ(:,4));
    
    original_bullet_norm = pointCloud(XYZ_norm(:,1:3), 'Intensity', bullet_surf_norm);
    growth_bullet_norm = pointCloud(XYZ_norm(:,1:3), 'Intensity', XYZ(:,4));

    clear XYZ bullet_surf bullet_surf_norm sph_center
    
    if iter==1
        % Figure to show surface heatmap on bullet   
        titlename =  'Initial surface map on "bullet" shape';  
        figure('Name','Initial surface map on "bullet shape"','visible',visible)
        pcshow(original_bullet);
        hold off
        xlabel('X', 'FontSize', 16);
        ylabel('Y', 'FontSize', 16);
        zlabel('Z', 'FontSize', 16);
        axis equal;
        t = colorbar;
        t.FontSize = 18;
        t.Color = legend_text_color;
        ylabel(t, 'Distance to reference surface (um)')
        colormap(jet) 

        ax = gca(); 
        ax.FontSize = 18; 

        title (titlename, 'FontSize', 20);
        view([-40, 20])

        filename_png=[pwd '/' filepath '/Fig102-initial-surface-bullet.png'];
        saveas(gcf,filename_png);

        if savefigure==1
            filename_matlab=[pwd '/' filepath '/Fig102-initial-surface-bullet.fig'];
            savefig(filename_matlab);
        end              
        clear ax t filename_matlab filename_png titlename
    end
      
    % Figure to show growth heatmap on bullet 
    titlename =  strcat(name, ' growth map on "bullet" shape');  
    figure('Name','Growth map on "bullet shape"','visible',visible)
    pcshow(growth_bullet);
    hold off
    xlabel('X', 'FontSize', 16);
    ylabel('Y', 'FontSize', 16);
    zlabel('Z', 'FontSize', 16);
    axis equal;
    t = colorbar;
    t.FontSize = 18;
    t.Color = legend_text_color;
    ylabel(t, 'Magnitude of displacement (normalized to max)')
    colormap(parula)%colormap(flipud(autumn))

    ax = gca(); 
    ax.CLim = [0 1];
    ax.FontSize = 18; 

    title (titlename, 'FontSize', 20);
    view([-40, 20])

    filename_png=[pwd '/' filepath '/Fig103-' name '-growth-bullet.png'];
    saveas(gcf,filename_png);

    if savefigure==1
        filename_matlab=[pwd '/' filepath '/Fig103-' name '-growth-bullet.fig'];
        savefig(filename_matlab);
    end              
    clear ax t filename_matlab filename_png titlename
    
    if  iter==1
        % Figure to show surface heatmap on normalized bullet   
        titlename =  'Initial surface map on normalized "bullet" shape'; 
        figure('Name','Initial surface map on normalized "bullet shape"','visible',visible)
        pcshow(original_bullet_norm);
        hold off
        xlabel('X', 'FontSize', 16);
        ylabel('Y', 'FontSize', 16);
        zlabel('Z', 'FontSize', 16);
        axis equal;
        xlim([-1 1])
        ylim([-1 1])
        zlim([0 inf]) 
        t = colorbar;
        t.FontSize = 18;
        t.Color = legend_text_color;
        ylabel(t, 'Normalized distance to reference surface')
        colormap(jet) 

        ax = gca(); 
        ax.CLim = [-colorbar_limit colorbar_limit];
        ax.FontSize = 18; 

        title (titlename, 'FontSize', 20);
        view([-40, 20])

        filename_png=[pwd '/' filepath '/Fig104-initial-surface-bullet-norm.png'];
        saveas(gcf,filename_png);

        if savefigure==1
            filename_matlab=[pwd '/' filepath '/Fig104-initial-surface-bullet-norm.fig'];
            savefig(filename_matlab);
        end              
        clear ax t filename_matlab filename_png titlename
    end
      
    % Figure to growth heatmap on bullet 
    titlename =  strcat(name, ' growth map on normalized "bullet" shape');  
    figure('Name','Growth map on normalized "bullet shape"','visible',visible)
    pcshow(growth_bullet_norm);
    hold off
    xlabel('X', 'FontSize', 16);
    ylabel('Y', 'FontSize', 16);
    zlabel('Z', 'FontSize', 16);
    axis equal;
    xlim([-1 1])
    ylim([-1 1])
    zlim([0 inf]) 
    t = colorbar;
    t.FontSize = 18;
    t.Color = legend_text_color;
    ylabel(t, 'Magnitude of displacement (normalized to max)')
    colormap(parula)%colormap(flipud(autumn))

    ax = gca(); 
    ax.CLim = [0 1];
    ax.FontSize = 18; 

    title (titlename, 'FontSize', 20);
    view([-40, 20])

    filename_png=[pwd '/' filepath '/Fig105-' name '-growth-bullet-norm.png'];
    saveas(gcf,filename_png);

    if savefigure==1
        filename_matlab=[pwd '/' filepath '/Fig105-' name '-growth-bullet-norm.fig'];
        savefig(filename_matlab);
    end              
    clear ax t filename_matlab filename_png titlename
    
    
    % Flatten out heatmaps on normalized bullet shape 

    % "Peel" open cylinder: convert to polar coord
    [theta,~] = cart2pol(XYZ_cyl_norm(:,1),XYZ_cyl_norm(:,2));

    flat_cyl_norm = zeros(size(XYZ_cyl_norm,1),4);
    flat_cyl_norm(:,1) = theta;
    flat_cyl_norm(:,2) = XYZ_cyl_norm(:,3);
    flat_cyl_norm(:,3) = XYZ_cyl(:,4);
    flat_cyl_norm(:,4) = intensity_cyl_norm(:,1);

    clear theta XYZ_cyl_norm XYZ_cyl intensity_cyl_norm

    % Project hemisphere to flat surface: convert to spherical coord
    % Translate hemisphere to origin of coord (0,0,0)
    [azimuth,elevation_sph,~] = cart2sph(XYZ_sph_norm(:,1), ...
                                         XYZ_sph_norm(:,2), ...
                                         XYZ_sph_norm(:,3) - sph_center_norm(3));

    flat_sph_norm = zeros(size(XYZ_sph_norm,1),4);
    flat_sph_norm(:,1) = azimuth;
    flat_sph_norm(:,2) = elevation_sph; 
    flat_sph_norm(:,3) = XYZ_sph(:,4);
    flat_sph_norm(:,4) = intensity_sph_norm(:,1);

    % Translate back to original vertical position of sphere.
    flat_sph_norm(:,2) = flat_sph_norm(:,2) + sph_center_norm(3);

    % Add the two flattened parts together
    flat_array_norm = cat(1,flat_cyl_norm,flat_sph_norm);

    clear azimuth elevation_sph XYZ_sph_norm XYZ_sph intensity_sph_norm
    clear sph_center_norm
    
    if iter==1
        % Figure to show surface heatmap flattened out 
        titlename =  'Flattened out initial surface map';  
        figure('Name','Flattened out initial surface map','visible',visible)
            scatter(flat_array_norm(:,1),...
                    flat_array_norm(:,2),...
                    5, flat_array_norm(:,4), 'filled')
            hold on
            axis equal
            xlabel('reference surface perimeter', 'FontSize', 16)
            ylabel('proximal-distal direction', 'FontSize', 16)

            t = colorbar;
            t.FontSize = 18;
            ylabel(t, 'Normalized distance to reference surface')
            colormap(jet)

            ax = gca(); 
            ax.CLim = [-colorbar_limit colorbar_limit];
            ax.FontSize = 18; 

            title (titlename, 'FontSize', 20);

        filename_png=[pwd '/' filepath '/Fig106-initial-surface-flattened-heatmap.png'];
        saveas(gcf,filename_png);

        if savefigure==1
            filename_matlab=[pwd '/' filepath '/Fig106-initial-surface-flattened-heatmap.fig'];
            savefig(filename_matlab);
        end              
        clear ax t filename_matlab filename_png titlename
    end
    
    % Figure to show growth heatmap flattened out
    titlename =  strcat(name, ' flattened out growth map');  
    figure('Name','Flattened out growth map','visible',visible)
        scatter(flat_array_norm(:,1),...
                flat_array_norm(:,2),...
                5, flat_array_norm(:,3), 'filled')
        hold on
        axis equal
        xlabel('reference surface perimeter', 'FontSize', 16)
        ylabel('proximal-distal direction', 'FontSize', 16)

        t = colorbar;
        t.FontSize = 18;
        ylabel(t, 'Magnitude of displacement (normalized to max)')
        colormap(parula)%colormap(flipud(autumn))

        ax = gca(); 
        ax.CLim = [0 1];
        ax.FontSize = 18; 

        title (titlename, 'FontSize', 20);
    
    filename_png=[pwd '/' filepath '/Fig107-' name '-growth-flattened-heatmap.png'];
    saveas(gcf,filename_png);

    if savefigure==1
        filename_matlab=[pwd '/' filepath '/Fig107-' name '-growth-flattened-heatmap.fig'];
        savefig(filename_matlab);
    end              
    clear ax t filename_matlab filename_png titlename
    
    % Create continuous heatmap
    
    % Figure showing data point count
    titlename =  strcat(name, ' binned data point count');  
    figure('Name','Binned scatter plot showing data point count','visible',visible);
        binscatter(flat_array_norm(:,1), flat_array_norm(:,2),NumBins)
        axis equal
        xlabel('reference surface perimeter', 'FontSize', 16)
        ylabel('proximal-distal direction', 'FontSize', 16)

        t = colorbar;
        t.FontSize = 16;
        ylabel(t, 'bin counts')

        ax = gca(); 
        ax.FontSize = 16; 
        title (titlename, 'FontSize', 20);
        
    filename_png=[pwd '/' filepath '/Fig108-' name '-heatmap-point-count.png'];
    saveas(gcf,filename_png);

    if savefigure==1
        filename_matlab=[pwd '/' filepath '/Fig108-' name '-heatmap-point-count.fig'];
        savefig(filename_matlab);
    end              
    clear ax t filename_matlab filename_png titlename

    % Order data in array for x-column
    %flat_array_norm_sorted = sortrows(flat_array_norm);
    [N_x,edges_x] = histcounts(flat_array_norm(:,1),NumBins(1));
    [N_y,edges_y] = histcounts(flat_array_norm(:,2),NumBins(2));

    % Bin data according to x-column & then to y-column
    binned_data = cell(NumBins(1),NumBins(2));    
    binned_data_mean = zeros(NumBins(1), NumBins(2));

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
            binned_data_mean_surf(i,j) = mean(aux(:,4));
            binned_data_mean_growth(i,j) = mean(aux(:,3));
        end
    end
    clear i j N_y N_x aux subarray edges_x edges_y

    
   HeatmapMeanSurf   = fillmissing(binned_data_mean_surf,'linear');
   HeatmapMeanGrowth = fillmissing(binned_data_mean_growth,'linear');
   
   if iter==1
        % Figure to show surface continuous heatmap
        titlename =  'Flattened out initial surface map';  
        figure('Name','Flattened out initial surface map','visible',visible)
            hm = heatmap(flip(HeatmapMeanSurf'));
            colormap(jet)
            hm.ColorLimits = [-colorbar_limit colorbar_limit];
            hm.XLabel ='reference surface perimeter';
            hm.YLabel ='proximal-distal direction';
            hm.Title =titlename;

            ax = gca(); 
            ax.FontSize = 18; 
            % Remove tick labels from x and y axis
            ax.XDisplayLabels = nan(size(ax.XDisplayData));
            ax.YDisplayLabels = nan(size(ax.YDisplayData));
            grid off;

        filename_png=[pwd '/' filepath '/Fig109-initial-surface-continuous-heatmap.png'];
        saveas(gcf,filename_png);

        if savefigure==1
            filename_matlab=[pwd '/' filepath '/Fig109-initial-surface-continuous-heatmap.fig'];
            savefig(filename_matlab);
        end              
        clear ax t filename_matlab filename_png titlename
    end
    
    % Figure to show growth continuous heatmap
    titlename =  strcat(name, ' flattened out growth map');  
    figure('Name','Flattened out growth map','visible',visible)
        hm = heatmap(flip(HeatmapMeanGrowth'));
        colormap(parula)
        %colormap(brewermap([],'BuPu'));
        hm.ColorLimits = [0 1];
        hm.XLabel ='reference surface perimeter';
        hm.YLabel ='proximal-distal direction';
        hm.Title =titlename;

        ax = gca(); 
        ax.FontSize = 18; 
        % Remove tick labels from x and y axis
        ax.XDisplayLabels = nan(size(ax.XDisplayData));
        ax.YDisplayLabels = nan(size(ax.YDisplayData));
        grid off;
        
    filename_png=[pwd '/' filepath '/Fig110-' name '-growth-continuous-heatmap.png'];
    saveas(gcf,filename_png);

    if savefigure==1
        filename_matlab=[pwd '/' filepath '/Fig110-' name '-growth-continuous-heatmap.fig'];
        savefig(filename_matlab);
    end              
    clear ax t filename_matlab filename_png titlename
    
end
clear iter


%% Clear leftover variables
clear all