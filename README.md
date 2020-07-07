Readme file for CompLimb-biomech
==============================================

Overview
--------
The CompLimb code is a computational tool developed by Ester Comellas to elucidate the mechanobiological regulation of limb development. It is the main output of her [MSCA-IF-GF project](https://cordis.europa.eu/project/rcn/221578/factsheet/en). The development of this formulation began in Northeastern University (Boston, 2019) and will continue in the Universitat Politècnica de Catalunya (Barcelona, 2021).

CompLimb is split into three parts. The first part, CompLimb-biomech, is based on the biphasic nonlinear poro-viscoelastic formulation developed by Ester  Comellas and Jean-Paul Pelteret in the University of Erlangen-Nuremberg (Erlangen, 2016-2018) to reproduce brain tissue behaviour in response to biomechanical loading. It uses the [deal.II open source finite element library](https://www.dealii.org/), a C++ software library supporting the creation of finite element codes and an open community of users and developers. 

In CompLimb-biomech the biological tissue is modelled as a biphasic material consisting in a fluid-saturated nonlinear porous solid. Continuum growth is modelled via the multiplicative split of the deformation gradient tensor. The governing equations are linearised using automatic differentiation and solved monolithically for the unknown solid displacements and fluid pore pressure values.



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

