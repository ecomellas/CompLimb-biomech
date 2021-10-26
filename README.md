Readme file for CompLimb-biomech
==============================================

Overview
--------
The CompLimb code is a computational tool developed by Ester Comellas to elucidate the mechanobiological regulation of limb development. It is the main output of her [MSCA-IF-GF project](https://cordis.europa.eu/project/rcn/221578/factsheet/en). The formulation was developed in Northeastern University (Boston, 2019-2021) and Universitat Politècnica de Catalunya (Barcelona, 2021).

The first part of CompLimb, CompLimb-biomech, is based on the biphasic nonlinear poro-viscoelastic formulation developed by Ester  Comellas and Jean-Paul Pelteret in the University of Erlangen-Nuremberg (Erlangen, 2016-2018) to reproduce brain tissue behaviour in response to biomechanical loading. It uses the [deal.II open source finite element library](https://www.dealii.org/), a C++ software library supporting the creation of finite element codes and an open community of users and developers. 

In CompLimb-biomech the biological tissue is modelled as a biphasic material consisting in a fluid-saturated nonlinear porous solid. Continuum growth is modelled via the multiplicative split of the deformation gradient tensor. The governing equations are linearised using automatic differentiation and solved monolithically for the unknown solid displacements and fluid pore pressure values. 


Publications
--------
This code has been used in the following study:

Ester Comellas, Johanna E Farkas, Giona Kleinberg, Katlyn Lloyd, Thomas Mueller, Timothy J Duerr, Jose J Muñoz, James R Monaghan and Sandra J Shefelbine (2021), Local mechanical stimuli shape tissue growth in vertebrate joint morphogenesis, bioRxiv 2021.08.28.458034; doi: [10.1101/2021.08.28.458034](https://doi.org/10.1101/2021.08.28.458034).


About the CompLimb project
--------
Understanding the roles of motion and mechanotransduction in joint formation holds promise for the study and treatment of joint deformities in humans. Joint development has been widely studied in axolotls (Ambystoma mexicanum), as these animals regrow whole limbs throughout their life. Axolotl limbs are morphologically similar to human limbs and utilize the same biological rubrics as ontogenic growth. To draw from the therapeutic potential of these similarities, we propose to build a multi-scale multi-physics computational model for the prediction of vertebrate limb development. Our model will be based on in vivo data obtained using novel imaging techniques via NSF-funded experiments on axolotl limb growth, and will be utilised to determine the physical mechanisms of normal and pathological joint morphogenesis. To this end, in AIM 1 we will build a finite element model of growth at the tissue level to study how specific changes in limb motion regulate joint morphology. Next, in AIM 2 we will build a model of growth at the molecular level to determine how biochemical and biomechanical signalling pathways interact during normal and pathological joint development. Finally, in AIM 3 we will integrate both experimental and computational data from the different length scales into a single multi-scale mechanobiochemical model of vertebrate limb growth. A computational model that links the biomechanics and biochemistry of normal and pathological limb development at the subcellular, cellular and tissue scales is a powerful predictive tool. We envisage this tool will be utilised to optimise treatment therapies for joint deformities and better inform the preventive screening of congenital defects in humans.


 Funding sources
 --------
 This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No.841047 –  A Computational tool to elucidate the mechano-biological regulation of Limb development. Experiments used to calibrate and validate the computational models are funded by National Science Foundation Grant #1727518 –  Mechano-biology of Joint Morphogenesis: Manipulating Salamander Limbs.


Branches
---------------- 
### master
This is the latest stable version of the code.

### mech-growth-function
A branch to explore alternatives for the mechanical growth function. Initially, mechanical growth was made proportional to pore pressure, but here we look at different measures of seepage velocity (norm, divergence, etc.) to replace have this velocity as mechanical stimulus, instead of the pressure. The divergence of seepage velocity as driver of mechanical growth was incorporated into the master branch

### reduced-geom-troubleshooting
A branch to troubleshoot problems we were having when running the case with the idealized geometry of the humerus.


Instructions
---------------- 
To run this code, follow the instructions on the [deal.ii website](https://www.dealii.org/download.html) to download and install deal.ii. Then, from a terminal in your computer, navigate to where the complimb-biomech.cc and CMakeLists.txt files are, and type:
> cmake .

Now build the code by typing:

> make release

or

> make debug

Then, following instructions, recompile by typing:

> make

To run the examples provided, create a folder “run” and copy the parameters.prm file (and the humerus_mesh.inp if running the realistic geometry examples) in there. Now, type:

> mpirun -np 4 -wdir ./run ../complimb-biomech

Change 4 for desired number of processes to be used.

The content of the parameters.inp file is self-explanatory, however the most important aspects are briefly described below.

Type of geometry:
- **growing_muffin**: Initial test to ensure continuum growth was correctly implemented. It loosely reproduces the example in Fig 4 of Kuhl (2014), doi: 10.1016/j.jmbbm.2013.10.009.
- **trapped_turtle**: Initial test to ensure continuum growth was correctly implemented. It loosely reproduces the example in Fig 3 of Kuhl (2014), doi: 10.1016/j.jmbbm.2013.10.009.
- **cube_growth**: Initial test to ensure continuum growth was correctly implemented. It is a simple cube. Different boundary conditions on displacements and pressure are available.
- **idealised_humerus**: initial test to ensure continuum growth was correctly implemeted. Different pressure boundary conditions are available.
- **external_mesh_humerus**: Different pressure boundary conditions are available.

Type of continuum growth:
- **none**: no growth
- **morphogen**: morphogenetic growth, i.e. constant volumetric growth that increases with each timestep. Used in growing_muffin, trapped_turtle and cube_growth examples.
- **pressure**: growth is proportional to pore pressure. Should work with any example.
- **joint-pressure**: growth is a sum of a biological component and pore-pressure-driven mechanical component. Only works with humerus examples.
- **joint-div-vel**: growth is a sum of a biological component and divergence-of-the-fluid-velocity-driven mechanical component. Only works with humerus examples.

The pressure-driven mechanical growth rate is computed as "growth rate mech  <pore pressure> ^growth exponential mech". The biological growth rate in the joint examples is implemented as a third order polynomial function along the proximo-distal axis. This was done following the approach shown in Fig.3 of Heegaard, Beaupré & Carter (1999), doi: 10.1002/jor.1100170408. However, analysis of our experimental data revealed a constant chondrocyte density in the regenerating axolotl limbs, so we set all the terms in the polynomial to zero, except growth bio coeff 0 = 1, to have a constant biological growth rate throughout the whole tissue equal to "growth rate bio".
