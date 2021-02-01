/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2010 - 2020 by the deal.II authors and
 *                              Ester Comellas
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License v3.0  as published by the Free Software Foundation.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 */

/*  CompLimb-biomech
 *  Author: Ester Comellas
 *          Northeastern University and
 *          Universitat Politècnica de Catalunya, 2019
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// We start by including all the necessary deal.II header files and some C++
// related ones. They have been discussed in detail in previous tutorial
// programs, so you need only refer to past tutorials for details.

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/function_spherical.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_linear_operator.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/transformations.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <iostream>
#include <fstream>
#include <numeric>
#include <iomanip>


// We create a namespace for everything that relates to
// the nonlinear poro-viscoelastic formulation,
// and import all the deal.II function and class names into it:
namespace CompLimb
{
  using namespace dealii;

  // @sect3{Run-time parameters}
  //
  // Set up a ParameterHandler object to read in the parameter choices at run-time
  // introduced by the user through the file "parameters.prm"
  namespace Parameters
  {
  // @sect4{Finite Element system}
  // Here we specify the polynomial order used to approximate the solution,
  // both for the displacements and pressure unknowns.
  // The quadrature order should be adjusted accordingly.
    struct FESystem
    {
      unsigned int poly_degree_displ;
      unsigned int poly_degree_pore;
      unsigned int quad_points;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree displ", "2",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");

        prm.declare_entry("Polynomial degree pore", "1",
                          Patterns::Integer(0),
                          "Pore pressure system polynomial order");

        prm.declare_entry("Quadrature points", "3",
                          Patterns::Integer(0),
                          "Gauss quadrature points per space dimension, "
                          "exact for polynomials of degree 2n-1");
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree_displ = prm.get_integer("Polynomial degree displ");
        poly_degree_pore = prm.get_integer("Polynomial degree pore");
        quad_points = prm.get_integer("Quadrature points");
      }
      prm.leave_subsection();
    }

  // @sect4{Geometry}
  // These parameters are related to the geometry definition and mesh generation.
  // We select the type of problem to solve and introduce the desired load values.
    struct Geometry
    {
      std::string  geom_type;
      unsigned int global_refinement;
      double       scale;
      std::string  load_type;
      double       load;
      double       fluid_flow;
      double       drained_pressure;
      double       joint_length;
      double       joint_radius;
      double       radius_phi_min;
      double       radius_phi_max;
      double       radius_theta_min;
      double       radius_theta_max;
      double       radius_area_r;
      double       ulna_theta_min;
      double       ulna_theta_max;
      double       ulna_phi_min;
      double       ulna_phi_max;
      double       ulna_area_r;
      double       load_reduction;
      unsigned int num_cycles;
      unsigned int num_no_load_time_steps;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry("Geometry type", "idealised_humerus_partially_drained",
                           Patterns::Selection("growing_muffin"
                                               "|trapped_turtle"
                                               "|cube_confined_drained"
                                               "|cube_confined_undrained"
                                               "|cube_unconfined_drained"
                                               "|cube_unconfined_undrained"
                                               "|idealised_humerus_partially_drained"
                                               "|idealised_humerus_fully_drained"
                                               "|idealised_humerus_laterals_undrained"
                                               "|external_mesh_humerus_partially_drained"
                                               "|external_mesh_humerus_fully_drained"
                                               "|external_mesh_humerus_laterals_undrained"),
                              "Type of geometry used. ");

        prm.declare_entry("Global refinement", "1",
                          Patterns::Integer(0),
                          "Global refinement level");

        prm.declare_entry("Grid scale", "1.0",
                          Patterns::Double(0.0),
                          "Global grid scaling factor");

        prm.declare_entry("Load type", "pressure",
                          Patterns::Selection("pressure|displacement|none"),
                          "Type of loading");

        prm.declare_entry("Load value", "-7.5e+6",
                          Patterns::Double(),
                          "Loading value");

        prm.declare_entry("Fluid flow value", "0.0",
                          Patterns::Double(),
                          "Prescribed fluid flow. Not implemented in any example yet.");

        prm.declare_entry("Drained pressure", "0.0",
                          Patterns::Double(),
                          "Increase of pressure value at drained boundary w.r.t "
                          "the atmospheric pressure. Idealised humerus always considers "
                          "a drained pressure of zero and will ignore this value.");

        prm.declare_entry("Joint length", "1.75",
                           Patterns::Double(0,1e6),
                           "Joint rudiment length, only for humerus geometries."
                           "For external_mesh_humerus indicate length to impose BCs properly.");

        prm.declare_entry("Joint radius", "0.5",
                           Patterns::Double(0,1e6),
                           "Joint rudiment radius, only for humerus geometries.");

        prm.declare_entry("Radius phi min", "10.",
                          Patterns::Double(0,360),
                          "Initial polar angle (in degrees) of loading cycle "
                          "corresponding to the effect of the radius, "
                          "only for humerus geometries.");

        prm.declare_entry("Radius phi max", "80.",
                          Patterns::Double(0,360),
                          "Final polar angle (in degrees) of loading cycle "
                          "corresponding to the effect of the radius, "
                          "only for humerus geometries.");

          prm.declare_entry("Radius theta min", "90.",
                            Patterns::Double(0,360),
                           "Initial polar angle (in degrees) of loading cycle "
                           "corresponding to the effect of the radius, "
                           "only for humerus geometries.");

          prm.declare_entry("Radius theta max", "90.",
                            Patterns::Double(0,360),
                            "Final polar angle (in degrees) of loading cycle "
                            "corresponding to the effect of the radius, "
                            "only for humerus geometries.");

         prm.declare_entry("Radius area radius", "0.25",
                           Patterns::Double(0,1e6),
                           "Radius of load contact area corresponding to "
                           "the effect of the radius, only for idealised_humerus.");

         prm.declare_entry("Ulna phi min", "40.",
                           Patterns::Double(0,360),
                           "Initial polar angle (in degrees) of loading cycle "
                           "corresponding to the effect of the ulna, "
                           "only for humerus geometries.");

         prm.declare_entry("Ulna phi max", "0.",
                           Patterns::Double(0,360),
                           "Final polar angle (in degrees) of loading cycle "
                           "corresponding to the effect of the ulna, "
                           "only for humerus geometries.");

          prm.declare_entry("Ulna theta min", "90.",
                            Patterns::Double(0,360),
                           "Initial polar angle (in degrees) of loading cycle "
                           "corresponding to the effect of the ulna, "
                           "only for humerus geometries.");

          prm.declare_entry("Ulna theta max", "90.",
                            Patterns::Double(0,360),
                            "Final polar angle (in degrees) of loading cycle "
                            "corresponding to the effect of the ulna, "
                            "only for humerus geometries.");

          prm.declare_entry("Ulna area radius", "0.1",
                            Patterns::Double(0,1e6),
                            "Radius of load contact area corresponding to "
                            "the effect of the ulna, only for humerus geometries.");
          
          prm.declare_entry("Load value reduction", "0.0",
                            Patterns::Double(-1,1),
                            "Reduction in value of loading (0:no reduction; "
                            "1:full reduction, i.e. no load at max angles), "
                            "only for humerus geometries. ");

         prm.declare_entry("Number of cycles", "1",
                           Patterns::Integer(1,1e6),
                           "Number of loading cycles. Each cycle follows min-max-min "
                           "angles in sinusoidal form, only for idealised_humerus.");

         prm.declare_entry("Number of no-load time steps", "0",
                            Patterns::Integer(0,1e6),
                            "Number of time steps at the end of the simulation "
                            "without load.");
      }
      prm.leave_subsection();
    }

    void Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        geom_type = prm.get("Geometry type");
        global_refinement = prm.get_integer("Global refinement");
        scale = prm.get_double("Grid scale");
        load_type = prm.get("Load type");
        load = prm.get_double("Load value");
        fluid_flow = prm.get_double("Fluid flow value");
        drained_pressure = prm.get_double("Drained pressure");
        joint_length = prm.get_double("Joint length");
        joint_radius = prm.get_double("Joint radius");
        radius_phi_min = prm.get_double("Radius phi min");
        radius_phi_max = prm.get_double("Radius phi max");
        radius_theta_min = prm.get_double("Radius theta min");
        radius_theta_max = prm.get_double("Radius theta max");
        radius_area_r = prm.get_double("Radius area radius");
        ulna_phi_min = prm.get_double("Ulna phi min");
        ulna_phi_max = prm.get_double("Ulna phi max");
        ulna_theta_min = prm.get_double("Ulna theta min");
        ulna_theta_max = prm.get_double("Ulna theta max");
        ulna_area_r = prm.get_double("Ulna area radius");
        load_reduction = prm.get_double("Load value reduction");
        num_cycles = prm.get_integer("Number of cycles");
        num_no_load_time_steps = prm.get_integer("Number of no-load time steps");
      }
      prm.leave_subsection();
    }

// @sect4{Materials}

// Here we select the type of material for the solid component
// and define the corresponding material parameters.
// Then we define he fluid data, including the type of
// seepage velocity definition to use.
  struct Materials
  {
    std::string  mat_type;
    double lambda;
    double mu;
    double mu1_infty;
    double mu2_infty;
    double mu3_infty;
    double alpha1_infty;
    double alpha2_infty;
    double alpha3_infty;
    double mu1_mode_1;
    double mu2_mode_1;
    double mu3_mode_1;
    double alpha1_mode_1;
    double alpha2_mode_1;
    double alpha3_mode_1;
    double viscosity_mode_1;
    std::string growth_type;
    double growth_rate_mech;
    double growth_exponential_mech;
    double growth_rate_bio;
    double growth_bio_coeff_0;
    double growth_bio_coeff_1;
    double growth_bio_coeff_2;
    double growth_bio_coeff_3;
    std::string  fluid_type;
    double solid_vol_frac;
    double kappa_darcy;
    double init_intrinsic_perm;
    double viscosity_FR;
    double init_darcy_coef;
    double weight_FR;
    bool gravity_term;
    int gravity_direction;
    double gravity_value;
    double density_FR;
    double density_SR;
    enum SymmetricTensorEigenvectorMethod eigen_solver;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

  void Materials::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Material properties");
    {
      prm.declare_entry("material", "Neo-Hooke",
                        Patterns::Selection("Neo-Hooke|Ogden|visco-Ogden"),
                        "Type of material used in the problem");

      prm.declare_entry("lambda", "8.375e6",
                        Patterns::Double(0,1e100),
                        "First Lamé parameter for extension function related "
                        "to compactation point in solid material [Pa].");

      prm.declare_entry("shear modulus", "5.583e6",
                        Patterns::Double(0,1e100),
                        "shear modulus for Neo-Hooke materials [Pa].");

      prm.declare_entry("eigen solver", "QL Implicit Shifts",
                        Patterns::Selection("QL Implicit Shifts|Jacobi"),
                        "The type of eigen solver to be used for Ogden and "
                        "visco-Ogden models.");

      prm.declare_entry("mu1", "0.0",
                        Patterns::Double(),
                        "Shear material parameter 'mu1' for Ogden material [Pa].");

      prm.declare_entry("mu2", "0.0",
                        Patterns::Double(),
                        "Shear material parameter 'mu2' for Ogden material [Pa].");

      prm.declare_entry("mu3", "0.0",
                        Patterns::Double(),
                        "Shear material parameter 'mu1' for Ogden material [Pa].");

      prm.declare_entry("alpha1", "1.0",
                        Patterns::Double(),
                        "Stiffness material parameter 'alpha1' for Ogden material [-].");

      prm.declare_entry("alpha2", "1.0",
                        Patterns::Double(),
                        "Stiffness material parameter 'alpha2' for Ogden material [-].");

      prm.declare_entry("alpha3", "1.0",
                        Patterns::Double(),
                        "Stiffness material parameter 'alpha3' for Ogden material [-].");

      prm.declare_entry("mu1_1", "0.0",
                        Patterns::Double(),
                        "Shear material parameter 'mu1' for first viscous mode "
                        "in Ogden material [Pa].");

      prm.declare_entry("mu2_1", "0.0",
                        Patterns::Double(),
                        "Shear material parameter 'mu2' for first viscous mode "
                        "in Ogden material [Pa].");

      prm.declare_entry("mu3_1", "0.0",
                        Patterns::Double(),
                        "Shear material parameter 'mu1' for first viscous mode "
                        "in Ogden material [Pa].");

      prm.declare_entry("alpha1_1", "1.0",
                        Patterns::Double(),
                        "Stiffness material parameter 'alpha1' for "
                        "first viscous mode in Ogden material [-].");

      prm.declare_entry("alpha2_1", "1.0",
                        Patterns::Double(),
                        "Stiffness material parameter 'alpha2' for "
                        "first viscous mode in Ogden material [-].");

      prm.declare_entry("alpha3_1", "1.0",
                        Patterns::Double(),
                        "Stiffness material parameter 'alpha3' for "
                        "first viscous mode in Ogden material [-].");

      prm.declare_entry("viscosity_1", "1e-10",
                        Patterns::Double(1e-10,1e100),
                        "Deformation-independent viscosity parameter "
                        "'eta_1' for first viscous mode in Ogden material [-].");

      prm.declare_entry("growth", "none",
                         Patterns::Selection("none|morphogen|pressure|joint-pressure|joint-div-vel"),
                         "Type of continuum growth");

      prm.declare_entry("growth incr", "1.0",
                        Patterns::Double(0,1e6),
                        "Morphogenetic growth increment per timestep");

      prm.declare_entry("growth rate mech", "0.01",
                        Patterns::Double(0,1e6),
                        "Growth rate for mechanically-stimulated growth");

      prm.declare_entry("growth exponential mech", "1.0",
                        Patterns::Double(1e-6,1e+6),
                        "Growth exponential that determines nature of the "
                        "dependency of growth on mechanical stimulus (linear, quadratic,"
                        " etc.) for mechanically-stimulated growth");

      prm.declare_entry("growth rate bio", "1e3",
                        Patterns::Double(0,1e6),
                        "Biological growth rate (based on chondrocyte density) "
                        "for joint growth.");

      prm.declare_entry("growth bio coeff 0", "0.14",
                        Patterns::Double(),
                        "Coefficient for order 0 term of polynomial defining "
                        "biological growth (based on chondrocyte density) "
                        "for joint growth.");

      prm.declare_entry("growth bio coeff 1", "-0.87",
                        Patterns::Double(),
                        "Coefficient for order 1 term of polynomial defining "
                        "biological growth (based on chondrocyte density) "
                        "for joint growth.");

      prm.declare_entry("growth bio coeff 2", "4.40",
                        Patterns::Double(),
                        "Coefficient for order 2 term of polynomial defining "
                        "biological growth (based on chondrocyte density) "
                        "for joint growth.");

      prm.declare_entry("growth bio coeff 3", "-2.66",
                        Patterns::Double(),
                        "Coefficient for order 3 term of polynomial defining "
                        "biological growth (based on chondrocyte density) "
                        "for joint growth.");

      prm.declare_entry("seepage definition", "Ehlers",
                        Patterns::Selection("Markert|Ehlers"),
                        "Type of formulation used to define the seepage velocity "
                        "in the problem. Choose between Markert formulation "
                        "of deformation-dependent intrinsic permeability "
                        "and Ehlers formulation of deformation-dependent "
                        "Darcy flow coefficient.");

      prm.declare_entry("initial solid volume fraction", "0.67",
                        Patterns::Double(0.001,0.999),
                        "Initial porosity (solid volume fraction, 0<n_0s<1)");

      prm.declare_entry("kappa", "0.0",
                        Patterns::Double(0,100),
                        "Deformation-dependency control parameter for "
                        "specific permeability (kappa >= 0)");

      prm.declare_entry("initial intrinsic permeability", "0.0",
                        Patterns::Double(0,1e100),
                        "Initial intrinsic permeability parameter [m^2] "
                        "(isotropic permeability). To be used with Markert formulation.");

      prm.declare_entry("fluid viscosity", "0.0",
                        Patterns::Double(0, 1e100),
                        "Effective shear viscosity parameter of the fluid "
                        "[Pa·s, (N·s)/m^2]. To be used with Markert formulation.");

      prm.declare_entry("initial Darcy coefficient", "1.0e-4",
                        Patterns::Double(0,1e100),
                        "Initial Darcy flow coefficient [m/s] (isotropic "
                        "permeability). To be used with Ehlers formulation.");

      prm.declare_entry("fluid weight", "1.0e4",
                        Patterns::Double(0, 1e100),
                        "Effective weight of the fluid [N/m^3]. "
                        "To be used with Ehlers formulation.");

      prm.declare_entry("gravity term", "false",
                        Patterns::Bool(),
                        "Gravity term considered (true) or neglected (false)");

      prm.declare_entry("fluid density", "1.0",
                        Patterns::Double(0,1e100),
                        "Real (or effective) density of the fluid");

      prm.declare_entry("solid density", "1.0",
                        Patterns::Double(0,1e100),
                        "Real (or effective) density of the solid");

      prm.declare_entry("gravity direction", "2",
                        Patterns::Integer(0,2),
                        "Direction of gravity (unit vector 0 for x, 1 for y, 2 for z)");

      prm.declare_entry("gravity value", "-9.81",
                        Patterns::Double(),
                        "Value of gravity (be careful to have consistent units!)");
    }
    prm.leave_subsection();
  }

  void Materials::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Material properties");
    {
      //Solid
      mat_type = prm.get("material");
      lambda = prm.get_double("lambda");
      mu = prm.get_double("shear modulus");
      mu1_infty = prm.get_double("mu1");
      mu2_infty = prm.get_double("mu2");
      mu3_infty = prm.get_double("mu3");
      alpha1_infty = prm.get_double("alpha1");
      alpha2_infty = prm.get_double("alpha2");
      alpha3_infty = prm.get_double("alpha3");
      mu1_mode_1 = prm.get_double("mu1_1");
      mu2_mode_1 = prm.get_double("mu2_1");
      mu3_mode_1 = prm.get_double("mu3_1");
      alpha1_mode_1 = prm.get_double("alpha1_1");
      alpha2_mode_1 = prm.get_double("alpha2_1");
      alpha3_mode_1 = prm.get_double("alpha3_1");
      viscosity_mode_1 = prm.get_double("viscosity_1");
      growth_type =  prm.get("growth");
      if ( growth_type == "morphogen" )
      {
        growth_rate_mech =  prm.get_double("growth incr");
        growth_exponential_mech = 0.0;
        growth_rate_bio = 0.0;
        growth_bio_coeff_0 = 0.0;
        growth_bio_coeff_1 = 0.0;
        growth_bio_coeff_2 = 0.0;
        growth_bio_coeff_3 = 0.0;
      }
      else if ( growth_type == "pressure" )
      {
        growth_rate_mech =  prm.get_double("growth rate mech");
        growth_exponential_mech =  prm.get_double("growth exponential mech");
        growth_rate_bio = 0.0;
        growth_bio_coeff_0 = 0.0;
        growth_bio_coeff_1 = 0.0;
        growth_bio_coeff_2 = 0.0;
        growth_bio_coeff_3 = 0.0;
      }

      else if ( (growth_type == "joint-pressure")||(growth_type == "joint-div-vel"))
      {
        growth_rate_mech =  prm.get_double("growth rate mech");
        growth_exponential_mech =  prm.get_double("growth exponential mech");
        growth_rate_bio =  prm.get_double("growth rate bio");
        growth_bio_coeff_0 = prm.get_double("growth bio coeff 0");
        growth_bio_coeff_1 = prm.get_double("growth bio coeff 1");
        growth_bio_coeff_2 = prm.get_double("growth bio coeff 2");
        growth_bio_coeff_3 = prm.get_double("growth bio coeff 3");
      }
      else
      {
        growth_rate_mech = 0.0;
        growth_exponential_mech = 0.0;
        growth_rate_bio = 0.0;
        growth_bio_coeff_0 = 0.0;
        growth_bio_coeff_1 = 0.0;
        growth_bio_coeff_2 = 0.0;
        growth_bio_coeff_3 = 0.0;
      }
      //Fluid
      fluid_type = prm.get("seepage definition");
      solid_vol_frac = prm.get_double("initial solid volume fraction");
      kappa_darcy = prm.get_double("kappa");
      init_intrinsic_perm = prm.get_double("initial intrinsic permeability");
      viscosity_FR = prm.get_double("fluid viscosity");
      init_darcy_coef = prm.get_double("initial Darcy coefficient");
      weight_FR = prm.get_double("fluid weight");
      //Gravity effects
      gravity_term = prm.get_bool("gravity term");
      density_FR = prm.get_double("fluid density");
      density_SR = prm.get_double("solid density");
      gravity_direction = prm.get_integer("gravity direction");
      gravity_value = prm.get_double("gravity value");

      if ( (fluid_type == "Markert") && ((init_intrinsic_perm == 0.0) ||
                                         (viscosity_FR == 0.0))           )
          AssertThrow(false, ExcMessage("Markert seepage velocity formulation "
          "requires the definition of 'initial intrinsic permeability' "
          "and 'fluid viscosity' greater than 0.0."));

      if ( (fluid_type == "Ehlers") && ((init_darcy_coef == 0.0) ||
                                        (weight_FR == 0.0))        )
          AssertThrow(false, ExcMessage("Ehler seepage velocity formulation "
          "requires the definition of 'initial Darcy coefficient' and "
          "'fluid weight' greater than 0.0."));

      const std::string eigen_solver_type = prm.get("eigen solver");
      if (eigen_solver_type == "QL Implicit Shifts")
        eigen_solver = SymmetricTensorEigenvectorMethod::ql_implicit_shifts;
      else if (eigen_solver_type == "Jacobi")
        eigen_solver = SymmetricTensorEigenvectorMethod::jacobi;
      else
      {
        AssertThrow(false, ExcMessage("Unknown eigen solver selected."));
      }
    }
    prm.leave_subsection();
  }

// @sect4{Nonlinear solver}
// We now define the tolerances and the maximum number of iterations for the
// Newton-Raphson scheme used to solve the nonlinear system of governing equations.
  struct NonlinearSolver
  {
    unsigned int max_iterations_NR;
    double       tol_f;
    double       tol_u;
    double       tol_p_fluid;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

  void NonlinearSolver::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Nonlinear solver");
    {
      prm.declare_entry("Max iterations Newton-Raphson", "15",
                        Patterns::Integer(0),
                        "Number of Newton-Raphson iterations allowed");

      prm.declare_entry("Tolerance force", "1.0e-8",
                        Patterns::Double(0.0),
                        "Force residual tolerance");

      prm.declare_entry("Tolerance displacement", "1.0e-6",
                        Patterns::Double(0.0),
                        "Displacement error tolerance");

      prm.declare_entry("Tolerance pore pressure", "1.0e-6",
                        Patterns::Double(0.0),
                        "Pore pressure error tolerance");
    }
    prm.leave_subsection();
  }

  void NonlinearSolver::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Nonlinear solver");
    {
      max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
      tol_f = prm.get_double("Tolerance force");
      tol_u = prm.get_double("Tolerance displacement");
      tol_p_fluid =  prm.get_double("Tolerance pore pressure");
    }
    prm.leave_subsection();
  }

// @sect4{Time}
// Here we set the timestep size $ \varDelta t $ and the simulation end-time.
  struct Time
  {
    double end_time;
    double delta_t;
    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

  void Time::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Time");
    {
      prm.declare_entry("End time", "10.0",
                        Patterns::Double(),
                        "End time");

      prm.declare_entry("Time step size", "0.002",
                        Patterns::Double(1.0e-6),
                        "Time step size. The value must be larger than the "
                        "displacement error tolerance defined.");
    }
    prm.leave_subsection();
  }

  void Time::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Time");
    {
      end_time = prm.get_double("End time");
      delta_t = prm.get_double("Time step size");
    }
    prm.leave_subsection();
  }


// @sect4{Output}
// We can choose the frequency of the data for the output files.
  struct OutputParam
  {
    std::string  outfiles_requested;
    unsigned int timestep_output;
    unsigned int poly_degree_output;
    std::string  outtype;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

  void OutputParam::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Output parameters");
    {
      prm.declare_entry("Output files", "all",
                        Patterns::Selection("none|bcs|solution|all"),
                        "Paraview output files to generate.");
      prm.declare_entry("Time step number output", "1",
                        Patterns::Integer(0),
                        "Output data for time steps multiple of the given "
                        "integer value.");
      prm.declare_entry("Output element order", "0",
                        Patterns::Integer(0),
                        "Order of elements for output. If undefined, "
                        "the same as the polynomial degree displ will be used.");
      prm.declare_entry("Averaged results", "nodes",
                         Patterns::Selection("elements|nodes"),
                         "Output data associated with integration point values"
                         " averaged on elements or on nodes.");
    }
    prm.leave_subsection();
  }

  void OutputParam::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Output parameters");
    {
      outfiles_requested = prm.get("Output files");
      timestep_output = prm.get_integer("Time step number output");
      poly_degree_output = prm.get_integer("Output element order");
      outtype = prm.get("Averaged results");
    }
    prm.leave_subsection();
  }

// @sect4{All parameters}
// We finally consolidate all of the above structures into a single container
// that holds all the run-time selections.
  struct AllParameters : public FESystem,
                         public Geometry,
                         public Materials,
                         public NonlinearSolver,
                         public Time,
                         public OutputParam
  {
    AllParameters(const std::string &input_file);

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

  AllParameters::AllParameters(const std::string &input_file)
  {
    ParameterHandler prm;
    declare_parameters(prm);
    prm.parse_input(input_file);
    parse_parameters(prm);
  }

  void AllParameters::declare_parameters(ParameterHandler &prm)
  {
    FESystem::declare_parameters(prm);
    Geometry::declare_parameters(prm);
    Materials::declare_parameters(prm);
    NonlinearSolver::declare_parameters(prm);
    Time::declare_parameters(prm);
    OutputParam::declare_parameters(prm);
  }

  void AllParameters::parse_parameters(ParameterHandler &prm)
  {
    FESystem::parse_parameters(prm);
    Geometry::parse_parameters(prm);
    Materials::parse_parameters(prm);
    NonlinearSolver::parse_parameters(prm);
    Time::parse_parameters(prm);
    OutputParam::parse_parameters(prm);
  }
}

// @sect3{Time class}
// A simple class to store time data.
// For simplicity we assume a constant time step size.
class Time
{
    public:
      Time (const double time_end,
            const double delta_t)
        :
        timestep(0),
        time_current(0.0),
        time_end(time_end),
        delta_t(delta_t)
      {}

      virtual ~Time()
      {}

      double get_current() const
      {
        return time_current;
      }
      double get_end() const
      {
        return time_end;
      }
      double get_delta_t() const
      {
        return delta_t;
      }
      unsigned int get_timestep() const
      {
        return timestep;
      }
      void increment_time ()
      {
        time_current += delta_t;
        ++timestep;
      }

    private:
      unsigned int timestep;
      double time_current;
      double time_end;
      const double delta_t;
};

// @sect3{Constitutive equation for the solid component of the biphasic material}

//@sect4{Base class: generic hyperelastic material}
// The ``extra" Kirchhoff stress in the solid component is the sum of isochoric
// and a volumetric part.
// $\mathbf{\tau} = \mathbf{\tau}_E^{(\bullet)} + \mathbf{\tau}^{\textrm{vol}}$
// The deviatoric part changes depending on the type of material model selected:
// Neo-Hooken hyperelasticity, Ogden hyperelasticiy,
// or a single-mode finite viscoelasticity based on the Ogden hyperelastic model.
// In this base class we declare  it as a virtual function,
// and it will be defined for each model type in the corresponding derived class.
// We define here the volumetric component, which depends on the
// extension function $U(J_S)$ selected, and in this case is the same for all models.
// We use the function proposed by
// Ehlers & Eipper 1999 doi:10.1023/A:1006565509095
// We also define some public functions to access and update the internal variables.
template <int dim, typename NumberType = Sacado::Fad::DFad<double> >
class Material_Hyperelastic
{
  public:
    Material_Hyperelastic(const Parameters::AllParameters &parameters,
                          const Time                      &time)
          :
          n_OS (parameters.solid_vol_frac),
          lambda (parameters.lambda),
          growth_type(parameters.growth_type),
          growth_rate_mech(parameters.growth_rate_mech),
          growth_exponential_mech(parameters.growth_exponential_mech),
          growth_rate_bio(parameters.growth_rate_bio),
          growth_bio_coeff_0(parameters.growth_bio_coeff_0),
          growth_bio_coeff_1(parameters.growth_bio_coeff_1),
          growth_bio_coeff_2(parameters.growth_bio_coeff_2),
          growth_bio_coeff_3(parameters.growth_bio_coeff_3),
          joint_length(parameters.joint_length),
          joint_radius(parameters.joint_radius),
          joint_num_no_load_timesteps(parameters.num_no_load_time_steps),
          time(time),
          growth_stretch(1.0),
          growth_stretch_converged(1.0),
          det_Fve (1.0),
          det_Fve_converged (1.0),
          eigen_solver (parameters.eigen_solver)
          {}
          ~Material_Hyperelastic()
          {}

    // Determine "extra" Kirchhoff stress as sum of isochoric and volumetric Kirchhoff stresses
    SymmetricTensor<2,dim,NumberType>
     get_tau_E(const Tensor<2,dim,NumberType> &F) const
     {
      //Compute (visco-elastic) part of the def. gradient tensor.
      const Tensor<2,dim> Fg = get_non_converged_growth_tensor();
      const Tensor<2,dim> Fg_inv = invert(Fg);
      const Tensor<2,dim,NumberType> Fve = F*Fg_inv;

      //should it be get_tau_E_ext_func(F)?!?!
      return ( get_tau_E_base(Fve) + get_tau_E_ext_func(Fve) );
     }

    // Determine "extra" Cauchy stress as Kirchhoff stresses
    SymmetricTensor<2,dim,NumberType>
     get_Cauchy_E(const Tensor<2,dim,NumberType> &F) const
     {
        const NumberType det_F = determinant(F);
        Assert(det_F>0, ExcInternalError());

        //Compute Fve
        const Tensor<2,dim> Fg = get_non_converged_growth_tensor();
        const Tensor<2,dim> Fg_inv = invert(Fg);
        const Tensor<2,dim,NumberType> Fve = F*Fg_inv;

        //Here the "whole" F is needed
        return (get_tau_E(Fve)*NumberType(1/det_F));
     }

    // Retrieve stored det_Fve
    double get_converged_det_Fve() const
    {
        return  det_Fve_converged;
    }

    double get_converged_growth_stretch() const
    {
        return growth_stretch_converged;
    }

    Tensor<2,dim> get_non_converged_growth_tensor() const
    {
        //Isotropic growth tensor
        Tensor<2,dim> Fg(Physics::Elasticity::StandardTensors<dim>::I);
        double theta = this->get_non_converged_growth_stretch();
        Fg *= theta;
        return Fg;
    }

    virtual void update_end_timestep()
    {
        det_Fve_converged = det_Fve;
        growth_stretch_converged = growth_stretch;
    }

    virtual void update_internal_equilibrium(const Tensor<2,dim,NumberType> &F,
                                             const NumberType &p_fluid_AD,
                                             const NumberType &div_seepage_vel,
                                             const Point<dim> &pt              )
    {
        const double det_F = Tensor<0,dim,double>(determinant(F));
        Tensor<2,dim> Fg = get_non_converged_growth_tensor();
        double det_Fg = determinant(Fg);
        det_Fve = det_F/det_Fg;

        //Growth
//        const double p_fluid = Tensor<0,dim,double>(p_fluid_AD);
        double mech_growth_stimulus;

        //No mechanical stimulus
        if ((growth_type == "none")||(growth_type == "morphogen"))
            mech_growth_stimulus = 0.0;

        //Mechanical stimulus is pressure
        else if ((growth_type == "pressure")||(growth_type == "joint-pressure"))
            mech_growth_stimulus = Tensor<0,dim,double>(p_fluid_AD);

        //Mechanical stimulus is divergence if seepage velocity
        else if (growth_type == "joint-div-vel")
            mech_growth_stimulus = Tensor<0,dim,double>(div_seepage_vel);

        else
            AssertThrow(false, ExcMessage("Growth type not implemented yet."));

        this->update_growth_stretch(mech_growth_stimulus,pt);
    }

    virtual double get_viscous_dissipation( ) const = 0;

    // Define constitutive model parameters
    const double n_OS; //Initial porosity (solid volume fraction)
    const double lambda; //1st Lamé parameter (for extension function related to compactation point)
    const std::string growth_type;
    const double growth_rate_mech; //Growth rate. For morphogen growth, increment per timestep
    const double growth_exponential_mech;
    const double growth_rate_bio;
    const double growth_bio_coeff_0;
    const double growth_bio_coeff_1;
    const double growth_bio_coeff_2;
    const double growth_bio_coeff_3;
    const double joint_length;
    const double joint_radius;
    const unsigned int joint_num_no_load_timesteps;
    const Time  &time;

    //Internal variables
    double growth_stretch;           // Value of internal variable at this Newton step and timestep
    double growth_stretch_converged; // Value of internal variable at the previous timestep
    double det_Fve;           //Value in current iteration in current time step
    double det_Fve_converged; //Value from previous time step

    const enum SymmetricTensorEigenvectorMethod eigen_solver;

  protected:
    //Compute growth criterion
    double get_growth_criterion(const double     &mech_growth_stimulus,
                                const Point<dim> &pt      ) const
    {
        double growth_criterion;

        //No growth
        if (growth_type == "none")
            growth_criterion=0.0;

        //Morphogenetic growth: growth rate = growth increment and is const in every time step
        else if (growth_type == "morphogen")
            growth_criterion=growth_rate_mech;

        //Growth driven by pressure
        else if (growth_type == "pressure")
        {
            double tolerance = 1.0e-6;
            if (mech_growth_stimulus > tolerance) //Growth only for compressive pressures
              growth_criterion = growth_rate_mech*
                              std::pow(mech_growth_stimulus,growth_exponential_mech);
            else
              growth_criterion=0.0;
        }
        //Growth driven by pressure in joint
        else if (growth_type == "joint-pressure")
        {
            const double current_time = time.get_current();
            const double dt = time.get_delta_t();
            const double end_load_time = dt*joint_num_no_load_timesteps;
            const double final_load_time = time.get_end() - end_load_time;
            
            // No growth for first load step and no-load steps at end.
            if ((current_time <= dt) || (current_time > final_load_time))
                growth_criterion=0.0;
            else
            {
                //Biological part
                const double chi = (pt[dim-1] + joint_length - joint_radius)/
                                   joint_length;
                growth_criterion = growth_rate_bio
                                    *(growth_bio_coeff_0 +
                                      growth_bio_coeff_1*chi +
                                      growth_bio_coeff_2*chi*chi +
                                      growth_bio_coeff_3*chi*chi*chi);

                //Mechanical part
                double tolerance = 1.0e-6;
                if (mech_growth_stimulus > tolerance) //Growth only for compressive pressures
                  growth_criterion += growth_rate_mech*
                                  std::pow(mech_growth_stimulus, (1.0/growth_exponential_mech));
            }
        }
        // Growth driven by divergence of the seepage velocity in joint
        // Exactly the same as above, but kept it separate, in case we need to change the function
        else if (growth_type == "joint-div-vel")
        {
            const double current_time = time.get_current();
            const double dt = time.get_delta_t();
            const double end_load_time = dt*joint_num_no_load_timesteps;
            const double final_load_time = time.get_end() - end_load_time;
            
            // No growth for first load step and no-load steps at end.
            if ((current_time <= dt) || (current_time > final_load_time))
                growth_criterion=0.0;
            else
            {
                //Biological part
                const double chi = (pt[dim-1] + joint_length - joint_radius)/
                                   joint_length;
                growth_criterion = growth_rate_bio
                                    *(growth_bio_coeff_0 +
                                      growth_bio_coeff_1*chi +
                                      growth_bio_coeff_2*chi*chi +
                                      growth_bio_coeff_3*chi*chi*chi);

                //Velocity-driven part
                double tolerance = 1.0e-6;
                if (mech_growth_stimulus > tolerance) //Growth only for positive divergences
                  growth_criterion += growth_rate_mech*
                                  std::pow(mech_growth_stimulus, (1.0/growth_exponential_mech));
            }
        }
        else
            AssertThrow(false, ExcMessage("Growth type not implemented yet."));

        return growth_criterion;

    }

    //Compute limiting function for growth
    double get_growth_limiting_function(const double &growth_stretch) const
    {
        //Not implemented yet: silence compiler warnings
        (void)growth_stretch;
        return (1.0);
    }

    //Compute derivative of growth stretch rate
    //(=growth_limiting_function*growth_criterion) w.r.t. growth stretch
    double get_derivative_growth_stretch_rate(const double &growth_stretch) const
    {
        //Not implemented yet: silence compiler warnings
        (void)growth_stretch;
        return (0.0);
    }

    //Compute growth stretch
    void update_growth_stretch(const double &mech_growth_stimulus, const Point<dim> &pt)
    {
       double growth_criterion = this->get_growth_criterion(mech_growth_stimulus,pt);
       growth_stretch = growth_stretch_converged;

        //If there is growth, compute growth stretch
        if (growth_criterion != 0.0)
        {
            double growth_stretch_old = growth_stretch_converged;
            double growth_stretch_new = growth_stretch_converged;
            double dt = time.get_delta_t();

            double tolerance = 1.0e-6;
            double residual = tolerance*10.0;

            while(abs(residual)>tolerance)
            {
                double growth_limiting_function =
                    this->get_growth_limiting_function(growth_stretch_new);
                double d_growth_stretch_rate_d_growth_stretch =
                    this->get_derivative_growth_stretch_rate(growth_stretch_new);

                residual = growth_stretch_old - growth_stretch_new
                           + growth_limiting_function*growth_criterion*dt;
                double K = 1.0 - dt*d_growth_stretch_rate_d_growth_stretch;

                growth_stretch_old = growth_stretch_new;
                growth_stretch_new = growth_stretch_old + residual/K;
            }
            growth_stretch = growth_stretch_new;

            //For morphogenic growth it's easier and faster to just write:
            //growth_stretch = growth_stretch_converged + growth_criterion;
        }
    }

    double get_non_converged_growth_stretch() const
    {
        return growth_stretch;
    }

    double get_non_converged_dgrowth_stretch_dt() const
    {
        return (growth_stretch - growth_stretch_converged);
    }

    // Extension function for "extra" Kirchhoff stress
    // Ehlers & Eipper 1999, doi:10.1023/A:1006565509095 --  eqn. (33)
    SymmetricTensor<2,dim,NumberType>
     get_tau_E_ext_func(const Tensor<2,dim,NumberType> &F) const
     {
        const NumberType det_F = determinant(F);
        Assert(det_F>0, ExcInternalError());

        static const SymmetricTensor<2,dim,double>
          I (Physics::Elasticity::StandardTensors<dim>::I);
        return  NumberType(lambda * (1.0-n_OS)*(1.0-n_OS) *
                           (det_F/(1.0-n_OS) - det_F/(det_F-n_OS))) * I;
     }

    // Hyperelastic part of "extra" Kirchhoff stress (will be defined in each derived class)
    // Must use compressible formulation
    virtual SymmetricTensor<2,dim,NumberType>
    get_tau_E_base(const Tensor<2,dim,NumberType> &F) const = 0;
};

//@sect4{Derived class: Neo-Hookean hyperelastic material}
template <int dim, typename NumberType=Sacado::Fad::DFad<double>>
class NeoHooke : public Material_Hyperelastic <dim,NumberType>
{
  public:
    NeoHooke(const Parameters::AllParameters &parameters,
             const Time                      &time)
    :
    Material_Hyperelastic <dim,NumberType> (parameters,time),
    mu(parameters.mu)
    {}
    virtual ~NeoHooke()
    {}

     double get_viscous_dissipation() const
     {
         return 0.0;
     }

  protected:
    const double mu;

    // Hyperelastic part of "extra" Kirchhoff stress (compressible formulation)
    // Ehlers & Eipper 1999, doi:10.1023/A:1006565509095 -- eqn. (33)
    SymmetricTensor<2,dim,NumberType>
     get_tau_E_base(const Tensor<2,dim,NumberType> &Fve) const
     {
       static const SymmetricTensor<2,dim,double>
          I (Physics::Elasticity::StandardTensors<dim>::I);
       const bool use_standard_model = true;

       if (use_standard_model)
       {
         // Standard Neo-Hooke
         return ( mu * ( symmetrize(Fve*transpose(Fve)) - I ) );
       }
       else
       {
         // Neo-Hooke in terms of principal stretches
         const SymmetricTensor<2,dim,NumberType>
            Bve = symmetrize(Fve*transpose(Fve));
         const std::array<std::pair<NumberType,Tensor<1,dim,NumberType>>,dim>
            eigen_Bve = eigenvectors(Bve, this->eigen_solver);

         SymmetricTensor<2,dim,NumberType> Bve_ev;
         for (unsigned int d=0; d<dim; ++d)
           Bve_ev += eigen_Bve[d].first*symmetrize(outer_product(
                                   eigen_Bve[d].second,eigen_Bve[d].second));
          return ( mu*(Bve_ev-I) );
       }
     }
};

//@sect4{Derived class: Ogden hyperelastic material}
template <int dim, typename NumberType = Sacado::Fad::DFad<double> >
class Ogden : public Material_Hyperelastic < dim, NumberType >
{
  public:
    Ogden(const Parameters::AllParameters &parameters,
          const Time                      &time)
    :
    Material_Hyperelastic< dim, NumberType > (parameters,time),
    mu({parameters.mu1_infty,
        parameters.mu2_infty,
        parameters.mu3_infty}),
    alpha({parameters.alpha1_infty,
           parameters.alpha2_infty,
           parameters.alpha3_infty})
    {}
    virtual ~Ogden()
    {}

     double get_viscous_dissipation() const
     {
         return 0.0;
     }

  protected:
    std::vector<double> mu;
    std::vector<double> alpha;

    // Hyperelastic part of "extra" Kirchhoff stress (compressible formulation)
    // Using same term as Ehlers & Eipper 1999, doi:10.1023/A:1006565509095 -- eqn. (33)
    // to guarantee stress-free reference configuration [ 0.5 * sum(mu[i]*alpha[i]) * ln J ]
    SymmetricTensor<2,dim,NumberType>
     get_tau_E_base(const Tensor<2,dim,NumberType> &Fve) const
     {
     //left Cauchy-Green deformation tensor
      const SymmetricTensor<2,dim,NumberType>
         Bve = symmetrize(Fve*transpose(Fve));

      //Compute Eigenvalues and Eigenvectors
      const std::array<std::pair<NumberType,Tensor<1,dim,NumberType>>,dim>
        eigen_Bve = eigenvectors(Bve, this->eigen_solver);

      SymmetricTensor<2,dim,NumberType>  tau;
      static const SymmetricTensor<2,dim,double>
        I (Physics::Elasticity::StandardTensors<dim>::I);

      for (unsigned int i=0; i<3; ++i)
      {
          for (unsigned int A=0; A<dim; ++A)
          {
              SymmetricTensor<2,dim,NumberType>  tau_aux1 = symmetrize(
                     outer_product(eigen_Bve[A].second,eigen_Bve[A].second));
              tau_aux1 *= mu[i]*std::pow(eigen_Bve[A].first, (alpha[i]/2.) );
              tau += tau_aux1;
          }
          SymmetricTensor<2,dim,NumberType> tau_aux2(I);
          tau_aux2 *= mu[i];
          tau -= tau_aux2;
      }
      return tau;
     }
};

//@sect4{Derived class: Single-mode Ogden viscoelastic material}
// We use the finite viscoelastic model described in
// Reese & Govindjee (1998) doi:10.1016/S0020-7683(97)00217-5
// The algorithm for the implicit exponential time integration is given in
// Budday et al. (2017) doi: 10.1016/j.actbio.2017.06.024
template <int dim, typename NumberType = Sacado::Fad::DFad<double>>
class visco_Ogden : public Material_Hyperelastic <dim,NumberType>
{
  public:
    visco_Ogden(const Parameters::AllParameters &parameters,
                const Time                      &time)
        :
        Material_Hyperelastic< dim, NumberType > (parameters,time),
        mu_infty({parameters.mu1_infty,
                  parameters.mu2_infty,
                  parameters.mu3_infty}),
        alpha_infty({parameters.alpha1_infty,
                     parameters.alpha2_infty,
                     parameters.alpha3_infty}),
        mu_mode_1({parameters.mu1_mode_1,
                   parameters.mu2_mode_1,
                   parameters.mu3_mode_1}),
        alpha_mode_1({parameters.alpha1_mode_1,
                      parameters.alpha2_mode_1,
                      parameters.alpha3_mode_1}),
        viscosity_mode_1(parameters.viscosity_mode_1),
        Cinv_v_1(Physics::Elasticity::StandardTensors<dim>::I),
        Cinv_v_1_converged(Physics::Elasticity::StandardTensors<dim>::I)
        {}
        virtual ~visco_Ogden()
        {}

    void update_internal_equilibrium( const Tensor<2,dim,NumberType> &F,
                                      const NumberType &p_fluid,
                                      const NumberType &div_seepage_vel,
                                      const Point<dim> &pt                 )
    {
        Material_Hyperelastic<dim,NumberType>::
            update_internal_equilibrium(F, p_fluid, div_seepage_vel, pt);

        // Finite viscoelasticity following Reese & Govindjee (1998)
        // Algorithm for implicit exponential time integration
        // as described in Budday et al. (2017) doi: 10.1016/j.actbio.2017.06.024

        // Initialize viscous part of right Cauchy-Green deformation tensor
        this->Cinv_v_1 = this->Cinv_v_1_converged;

        //Just one Maxwell element, no for-loop needed
        //Elastic predictor step (trial values)

        //Compute Fve
        const Tensor<2,dim> Fg = this->get_non_converged_growth_tensor();
        const Tensor<2,dim> Fg_inv = invert(Fg);
        const Tensor<2,dim,NumberType> Fve = F*Fg_inv;

        //Trial elastic part of left Cauchy-Green deformation tensor
        SymmetricTensor<2,dim,NumberType>
            B_e_1_tr = symmetrize(Fve * (this->Cinv_v_1) * transpose(Fve));

        //Compute Eigenvalues and Eigenvectors
        const std::array<std::pair<NumberType,Tensor<1,dim,NumberType>>,dim>
          eigen_B_e_1_tr = eigenvectors(B_e_1_tr, this->eigen_solver);

        Tensor<1,dim,NumberType> lambdas_e_1_tr;
        Tensor<1,dim,NumberType> epsilon_e_1_tr;
        for (int a=0; a<dim; ++a)
        {
            //Trial elastic principal stretches
            lambdas_e_1_tr[a] = std::sqrt(eigen_B_e_1_tr[a].first);
            //Trial elastic logarithmic principal stretches
            epsilon_e_1_tr[a] = std::log(lambdas_e_1_tr[a]);
        }

       //Inelastic corrector step
       const double tolerance = 1e-8;
       double residual_check = tolerance*10.0;
       Tensor<1,dim,NumberType> residual;
       Tensor<2,dim,NumberType> tangent;
       static const SymmetricTensor<2,dim,double>
          I(Physics::Elasticity::StandardTensors<dim>::I);
       NumberType J_e_1 = std::sqrt(determinant(B_e_1_tr));

       std::vector<NumberType> lambdas_e_1_iso(dim);
       SymmetricTensor<2,dim,NumberType> B_e_1;
       int iteration = 0;

       Tensor<1,dim,NumberType> lambdas_e_1;
       Tensor<1,dim,NumberType> epsilon_e_1;
       epsilon_e_1 = epsilon_e_1_tr;

        while(residual_check > tolerance)
        {
          NumberType aux_J_e_1 = 1.0;
          for (unsigned int a=0; a<dim; ++a)
          {
            lambdas_e_1[a] = std::exp(epsilon_e_1[a]);
            aux_J_e_1 *= lambdas_e_1[a];
          }
          J_e_1 = aux_J_e_1;

          for (unsigned int a=0; a<dim; ++a)
              lambdas_e_1_iso[a] = lambdas_e_1[a]*std::pow(J_e_1,-1.0/dim);

          for (unsigned int a=0; a<dim; ++a)
          {
            residual[a] = get_beta_mode_1(lambdas_e_1_iso, a);
            residual[a] *= this->time.get_delta_t()/(2.0*viscosity_mode_1);
            residual[a] += epsilon_e_1[a];
            residual[a] -= epsilon_e_1_tr[a];

            for (unsigned int b=0; b<dim; ++b)
            {
              tangent[a][b]  = get_gamma_mode_1(lambdas_e_1_iso, a, b);
              tangent[a][b] *= this->time.get_delta_t()/(2.0*viscosity_mode_1);
              tangent[a][b] += I[a][b];
            }

            }
            //Update elastic logarithmic principal stretches
            epsilon_e_1 -= invert(tangent)*residual;
            //Update residual check
            residual_check = 0.0;
            for (unsigned int a=0; a<dim; ++a)
            {
              if (std::abs(residual[a])>residual_check)
                  residual_check = std::abs(Tensor<0,dim,double>(residual[a]));
            }
            iteration += 1;
            if (iteration>15)
              AssertThrow(false, ExcMessage("No convergence in local Newton iteration "
                                        "for the viscoelastic exponential time "
                                        "integration algorithm."));
        }

        //Compute converged stretches and left Cauchy-Green deformation tensor of mode 1
        NumberType aux_J_e_1 = 1.0;
        for (unsigned int a=0; a<dim; ++a)
        {
            lambdas_e_1[a] = std::exp(epsilon_e_1[a]);
            aux_J_e_1 *= lambdas_e_1[a];
        }
        J_e_1 = aux_J_e_1;

        for (unsigned int a=0; a<dim; ++a)
            lambdas_e_1_iso[a] = lambdas_e_1[a]*std::pow(J_e_1,-1.0/dim);

        for (unsigned int a=0; a<dim; ++a)
        {
            SymmetricTensor<2,dim,NumberType>
            B_e_1_aux = symmetrize(outer_product(
                          eigen_B_e_1_tr[a].second,eigen_B_e_1_tr[a].second));
            B_e_1_aux *= lambdas_e_1[a] * lambdas_e_1[a];
            B_e_1 += B_e_1_aux;
        }
        //Update inverse of the viscous right Cauchy-Green deformation tensor of mode 1
        Tensor<2,dim,NumberType> Cinv_v_1_AD = symmetrize(invert(F)
                                             * B_e_1 * invert(transpose(F)));
        //Update tau_E_neq_1
        this->tau_neq_1 = 0;
        for (unsigned int a=0; a<dim; ++a)
        {
            SymmetricTensor<2,dim,NumberType>
            tau_neq_1_aux = symmetrize(outer_product
                        (eigen_B_e_1_tr[a].second,eigen_B_e_1_tr[a].second));
            tau_neq_1_aux *=  get_beta_mode_1(lambdas_e_1_iso, a);
            this->tau_neq_1 += tau_neq_1_aux;
        }
        // Store history
        for (unsigned int a=0; a<dim; ++a)
            for (unsigned int b=0; b<dim; ++b)
                this->Cinv_v_1[a][b]= Tensor<0,dim,double>(Cinv_v_1_AD[a][b]);
    }

    void update_end_timestep()
    {
        Material_Hyperelastic <dim,NumberType>::update_end_timestep();
        this->Cinv_v_1_converged = this->Cinv_v_1;
    }

     double get_viscous_dissipation() const
     {                              //Double contract the two SymmetricTensor
         NumberType dissipation_term = get_tau_E_neq() * get_tau_E_neq();
         dissipation_term /= (2*viscosity_mode_1);
         return dissipation_term.val();
     }

  protected:
    std::vector<double> mu_infty;
    std::vector<double> alpha_infty;
    std::vector<double> mu_mode_1;
    std::vector<double> alpha_mode_1;
    double viscosity_mode_1;
    SymmetricTensor<2, dim, double> Cinv_v_1;
    SymmetricTensor<2, dim, double> Cinv_v_1_converged;
    SymmetricTensor<2, dim, NumberType> tau_neq_1;

    // Visco-hyperelastic part of "extra" Kirchhoff stress (compressible formulation)
    SymmetricTensor<2,dim,NumberType>
      get_tau_E_base(const Tensor<2,dim,NumberType> &Fve) const
      {
        return ( get_tau_E_neq() + get_tau_E_eq(Fve) );
      }

    // Equilibrium (hyperelastic) part of "extra" Kirchhoff stress
    SymmetricTensor<2, dim, NumberType>
      get_tau_E_eq(const Tensor<2,dim, NumberType> &Fve) const
      {
        //left Cauchy-Green deformation tensor
        const SymmetricTensor<2,dim,NumberType>
          Bve = symmetrize(Fve * transpose(Fve));

        //Compute Eigenvalues and Eigenvectors
        std::array<std::pair<NumberType,Tensor<1,dim,NumberType>>,dim> eigen_Bve;
        eigen_Bve = eigenvectors(Bve, this->eigen_solver);

        SymmetricTensor<2,dim,NumberType> tau;
        static const SymmetricTensor<2,dim,double>
          I (Physics::Elasticity::StandardTensors<dim>::I);

        for (unsigned int i=0; i<3; ++i)
        {
            for (unsigned int A=0; A<dim; ++A)
            {
                SymmetricTensor<2,dim,NumberType>  tau_aux1 = symmetrize(
                     outer_product(eigen_Bve[A].second,eigen_Bve[A].second));
                tau_aux1 *= mu_infty[i]*std::pow(eigen_Bve[A].first,
                                                 (alpha_infty[i]/2.) );
                tau += tau_aux1;
            }
            SymmetricTensor<2,dim,NumberType> tau_aux2(I);
            tau_aux2 *= mu_infty[i];
            tau -= tau_aux2;
        }
        return tau;
      }

    SymmetricTensor<2,dim,NumberType> get_tau_E_neq() const
    {
        return tau_neq_1;
    }

    //Compute beta term for the given (volume invariant) stretches
    NumberType get_beta_mode_1(std::vector<NumberType> &lambda_ve,
                               const int               &A         ) const
    {
        NumberType beta = 0.0;
        for (unsigned int i=0; i<3; ++i) //3rd-order Ogden model
        {
            NumberType aux = 0.0;
            for (int p=0; p<dim; ++p)
                aux += std::pow(lambda_ve[p],alpha_mode_1[i]);

            aux *= -1.0/dim;
            aux += std::pow(lambda_ve[A], alpha_mode_1[i]);
            aux *= mu_mode_1[i];

            beta  += aux;
        }
        return beta;
    }

    //Compute gamma term for the given (volume invariant) stretches
    NumberType get_gamma_mode_1(std::vector<NumberType> &lambda_ve,
                                const int               &A,
                                const int               &B          ) const
    {
        NumberType gamma = 0.0;
        if (A==B)
        {
            for (unsigned int i=0; i<3; ++i)
            {
                NumberType aux = 0.0;
                for (int p=0; p<dim; ++p)
                    aux += std::pow(lambda_ve[p],alpha_mode_1[i]);

                aux *= 1.0/(dim*dim);
                aux += 1.0/dim * std::pow(lambda_ve[A], alpha_mode_1[i]);
                aux *= mu_mode_1[i]*alpha_mode_1[i];

                gamma += aux;
            }
        }
        else
        {
            for (unsigned int i=0; i<3; ++i)
            {
                NumberType aux = 0.0;
                for (int p=0; p<dim; ++p)
                    aux += std::pow(lambda_ve[p],alpha_mode_1[i]);

                aux *= 1.0/(dim*dim);
                aux -= 1.0/dim * std::pow(lambda_ve[A], alpha_mode_1[i]);
                aux -= 1.0/dim * std::pow(lambda_ve[B], alpha_mode_1[i]);
                aux *= mu_mode_1[i]*alpha_mode_1[i];

                gamma += aux;
            }
        }
        return gamma;
    }
};


// @sect3{Constitutive equation for the fluid component of the biphasic material}
// We consider two slightly different definitions to define the seepage velocity with a Darcy-like law.
// Ehlers & Eipper 1999, doi:10.1023/A:1006565509095
// Markert 2007, doi:10.1007/s11242-007-9107-6
// The selection of one or another is made by the user via the parameters file.
template <int dim, typename NumberType = Sacado::Fad::DFad<double>>
class Material_Darcy_Fluid
{
   public:
      Material_Darcy_Fluid(const Parameters::AllParameters &parameters)
        :
        fluid_type(parameters.fluid_type),
        n_OS(parameters.solid_vol_frac),
        initial_intrinsic_permeability(parameters.init_intrinsic_perm),
        viscosity_FR(parameters.viscosity_FR),
        initial_darcy_coefficient(parameters.init_darcy_coef),
        weight_FR(parameters.weight_FR),
        kappa_darcy(parameters.kappa_darcy),
        gravity_term(parameters.gravity_term),
        density_FR(parameters.density_FR),
        gravity_direction(parameters.gravity_direction),
        gravity_value(parameters.gravity_value)
      {
        Assert(kappa_darcy >= 0, ExcInternalError());
      }
      ~Material_Darcy_Fluid()
      {}

     Tensor<1, dim, NumberType> get_seepage_velocity_current
                        (const Tensor<2,dim, NumberType> &F,
                         const Tensor<1,dim, NumberType> &grad_p_fluid) const
     {
         const NumberType det_F = determinant(F);
         Assert(det_F>0.0, ExcInternalError());
         Tensor<2,dim,NumberType> permeability_term;

         if (fluid_type == "Markert")
             permeability_term = get_instrinsic_permeability_current(F)/viscosity_FR;

         else if (fluid_type == "Ehlers")
             permeability_term = get_darcy_flow_current(F)/weight_FR;

         else
             AssertThrow(false, ExcMessage("Material_Darcy_Fluid --> Only Markert "
                            "and Ehlers formulations have been implemented."));

         return (-1.0 * permeability_term * det_F
                  * (grad_p_fluid - get_body_force_FR_current()) );
     }

    // Compute divergence of seepage velocity
    NumberType get_div_seepage_vel(const Tensor<1,dim,NumberType> &grad_p_fluid,
                                   const Tensor<2,dim,NumberType> &hess_p_fluid,
                                   const Tensor<1,dim,NumberType> &grad_det_F,
                                   const Tensor<2,dim,NumberType> &F) const
    {
        // Compute fluid material parameters required and do some checks
        const NumberType det_F = determinant(F);
        Assert(det_F>0.0, ExcInternalError());

        if (fluid_type != "Markert")
            AssertThrow(false, ExcMessage("Growth driven by divergence of seepage velocity "
                           "has been implemented for Markert formulation only."));

        static const SymmetricTensor<2,dim,double>
            I(Physics::Elasticity::StandardTensors<dim>::I);
        const Tensor<2,dim,NumberType> initial_instrinsic_permeability_tensor_T
                     = transpose(Tensor<2,dim,double>(initial_intrinsic_permeability*I));
        const Tensor<2,dim,NumberType> instrinsic_permeability_tensor_T
                     = transpose(get_instrinsic_permeability_current(F));
        const NumberType numerator = NumberType(std::pow( (det_F - n_OS), (kappa_darcy - 1.0)));
        const NumberType denominator = NumberType(std::pow( (1.0 - n_OS), kappa_darcy));
        const NumberType constvalue = kappa_darcy * numerator / denominator;
        const Tensor<1,dim,NumberType> func_grad_det_F = constvalue * grad_det_F;
        
        
        // Double contraction following Holzapfel notation, i.e.  A:B = A_ij B_ij
        NumberType div_seepage_vel = NumberType(
            (grad_p_fluid - get_body_force_FR_current())
               * (func_grad_det_F * initial_instrinsic_permeability_tensor_T))
             + double_contract<0,0,1,1>
                (instrinsic_permeability_tensor_T,hess_p_fluid);

        return (-1.0 / viscosity_FR * div_seepage_vel);
    }

     double get_porous_dissipation
                        (const Tensor<2,dim,NumberType> &F,
                         const Tensor<1,dim,NumberType> &grad_p_fluid) const
     {
         NumberType dissipation_term;
         Tensor<1,dim,NumberType> seepage_velocity;
         Tensor<2,dim,NumberType> permeability_term;

         const NumberType det_F = determinant(F);
         Assert(det_F>0.0, ExcInternalError());

         if (fluid_type == "Markert")
         {
             permeability_term = get_instrinsic_permeability_current(F)/viscosity_FR;
             seepage_velocity = get_seepage_velocity_current(F,grad_p_fluid);
         }
         else if (fluid_type == "Ehlers")
         {
             permeability_term = get_darcy_flow_current(F) / weight_FR;
             seepage_velocity = get_seepage_velocity_current(F,grad_p_fluid);
         }
         else
             AssertThrow(false, ExcMessage("Material_Darcy_Fluid --> Only Markert "
                               "and Ehlers formulations have been implemented."));

         dissipation_term = ( invert(permeability_term) * seepage_velocity )
                            * seepage_velocity;
         dissipation_term *= 1.0/(det_F*det_F);
         return Tensor<0,dim,double>(dissipation_term);
     }

   protected:
     const std::string  fluid_type;
     const double n_OS;
     const double initial_intrinsic_permeability;
     const double viscosity_FR;
     const double initial_darcy_coefficient;
     const double weight_FR;
     const double kappa_darcy;
     const bool   gravity_term;
     const double density_FR;
     const int    gravity_direction;
     const double gravity_value;

     Tensor<2, dim, NumberType>
      get_instrinsic_permeability_current(const Tensor<2,dim, NumberType> &F) const
      {
         static const SymmetricTensor<2,dim,double>
            I(Physics::Elasticity::StandardTensors<dim>::I);
         const Tensor<2,dim,NumberType> initial_instrinsic_permeability_tensor
                     = Tensor<2,dim,double>(initial_intrinsic_permeability*I);

         const NumberType det_F = determinant(F);
         Assert(det_F>0.0, ExcInternalError());

         const NumberType fraction = (det_F - n_OS)/(1 - n_OS);
         return ( NumberType (std::pow(fraction, kappa_darcy))
                  * initial_instrinsic_permeability_tensor);
      }

     Tensor<2,dim,NumberType>
        get_darcy_flow_current(const Tensor<2,dim,NumberType> &F) const
        {
           static const SymmetricTensor<2,dim,double>
              I(Physics::Elasticity::StandardTensors<dim>::I);
           const Tensor<2,dim,NumberType> initial_darcy_flow_tensor
                          = Tensor<2,dim,double>(initial_darcy_coefficient*I);

           const NumberType det_F = determinant(F);
           Assert(det_F>0.0, ExcInternalError());

           const NumberType fraction = (1.0 - (n_OS / det_F) )/(1.0 - n_OS);
           return ( NumberType (std::pow(fraction, kappa_darcy))
                    * initial_darcy_flow_tensor);
        }

    Tensor<1,dim,NumberType> get_body_force_FR_current() const
    {
        Tensor<1,dim,NumberType> body_force_FR_current;

        if (gravity_term == true)
        {
           Tensor<1,dim,NumberType> gravity_vector;
           gravity_vector[gravity_direction] = gravity_value;
           body_force_FR_current = density_FR * gravity_vector;
        }
        return body_force_FR_current;
    }
};

// @sect3{Quadrature point history}
// As seen in step-18, the <code> PointHistory </code> class offers a method
// for storing data at the quadrature points.  Here each quadrature point
// holds a pointer to a material description.  Thus, different material models
// can be used in different regions of the domain.  Among other data, we
// choose to store the ``extra" Kirchhoff stress $\boldsymbol{\tau}_E$ and
// the dissipation values $\mathcal{D}_p$ and $\mathcal{D}_v$.
template <int dim, typename NumberType = Sacado::Fad::DFad<double>> //double>
class PointHistory
{
    public:
        PointHistory()
        {}

        virtual ~PointHistory()
        {}

        void setup_lqp (const Parameters::AllParameters &parameters,
                        const Time                      &time)
        {

          if (parameters.mat_type == "Neo-Hooke")
              solid_material.reset(new NeoHooke<dim,NumberType>(parameters,time));
          else if (parameters.mat_type == "Ogden")
              solid_material.reset(new Ogden<dim,NumberType>(parameters,time));
          else if (parameters.mat_type == "visco-Ogden")
              solid_material.reset(new visco_Ogden<dim,NumberType>(parameters,time));
          else
              Assert (false, ExcMessage("Material type not implemented"));

          fluid_material.reset(new Material_Darcy_Fluid<dim,NumberType>(parameters));
        }

        // We offer an interface to retrieve certain data (used in the material and
        // global tangent matrix and residual assembly operations)
        SymmetricTensor<2,dim,NumberType>
          get_tau_E(const Tensor<2,dim,NumberType> &Fve) const
          {
            return solid_material->get_tau_E(Fve);
          }

        SymmetricTensor<2,dim,NumberType>
          get_Cauchy_E(const Tensor<2,dim,NumberType> &Fve) const
          {
            return solid_material->get_Cauchy_E(Fve);
          }

        double get_converged_det_Fve() const
        {
          return solid_material->get_converged_det_Fve();
        }

        double get_converged_growth_stretch() const
        {
          return solid_material->get_converged_growth_stretch();
        }

        Tensor<2,dim> get_non_converged_growth_tensor() const
        {
          return solid_material->get_non_converged_growth_tensor();
        }

        void update_end_timestep()
        {
          solid_material->update_end_timestep();
        }

        void update_internal_equilibrium(const Tensor<2,dim,NumberType> &F,
                                         const NumberType &p_fluid,
                                         const NumberType &div_seepage_vel,
                                         const Point<dim> &pt               )
        {
            solid_material->update_internal_equilibrium(F,p_fluid,div_seepage_vel,pt);
        }

        double get_viscous_dissipation() const
        {
            return solid_material->get_viscous_dissipation();
        }

        Tensor<1,dim,NumberType> get_seepage_velocity_current
                        (const Tensor<2,dim,NumberType> &F,
                         const Tensor<1,dim,NumberType> &grad_p_fluid) const
         {
             return fluid_material->get_seepage_velocity_current(F,grad_p_fluid);
         }

       NumberType get_div_seepage_vel( const Tensor<1,dim,NumberType> &grad_p_fluid,
                                       const Tensor<2,dim,NumberType> &hess_p_fluid,
                                       const Tensor<1,dim,NumberType> &grad_det_F,
                                       const Tensor<2,dim,NumberType> &F) const
        {
            return fluid_material->get_div_seepage_vel(grad_p_fluid,hess_p_fluid,grad_det_F,F);
        }

        double get_porous_dissipation
                (const Tensor<2,dim,NumberType> &F,
                 const Tensor<1,dim,NumberType> &grad_p_fluid) const
        {
            return fluid_material->get_porous_dissipation(F, grad_p_fluid);
        }

        Tensor<1,dim,NumberType> get_overall_body_force
                          (const Tensor<2,dim,NumberType>  &F,
                           const Parameters::AllParameters &parameters) const
        {
            Tensor<1,dim,NumberType> body_force;
            if (parameters.gravity_term == true)
            {
                const NumberType det_F_AD = determinant(F);
                Assert(det_F_AD>0.0, ExcInternalError());

                const NumberType overall_density_ref
                     = parameters.density_SR * parameters.solid_vol_frac
                      + parameters.density_FR
                      * (det_F_AD - parameters.solid_vol_frac);

               Tensor<1,dim,NumberType> gravity_vector;
               gravity_vector[parameters.gravity_direction] = parameters.gravity_value;
               body_force = overall_density_ref * gravity_vector;
            }
            return body_force;
        }
    private:
        std::shared_ptr<Material_Hyperelastic<dim,NumberType>> solid_material;
        std::shared_ptr<Material_Darcy_Fluid<dim,NumberType>> fluid_material;
};

// @sect3{Nonlinear poro-viscoelastic solid}
// The Solid class is the central class as it represents the problem at hand:
// the nonlinear poro-viscoelastic solid
template <int dim>
class Solid
{
      public:
        Solid(const Parameters::AllParameters &parameters);
        virtual ~Solid();
        void run();

      protected:
        using ADNumberType = Sacado::Fad::DFad<double>;

        std::ofstream outfile;
        std::ofstream pointfile;

        struct PerTaskData_ASM;
        template<typename NumberType = double> struct ScratchData_ASM;

        //Generate mesh
        virtual void make_grid() = 0;

        //Define points for post-processing
        virtual void define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices) = 0;

        //Set up the finite element system to be solved:
        void system_setup(TrilinosWrappers::MPI::BlockVector &solution_delta_OUT);

        //Extract sub-blocks from the global matrix
        void determine_component_extractors();

        // Several functions to assemble the system and
        // right hand side matrices using multithreading.
        void assemble_system
            (const TrilinosWrappers::MPI::BlockVector &solution_delta_OUT);
        void assemble_system_one_cell
            (const typename DoFHandler<dim>::active_cell_iterator &cell,
             ScratchData_ASM<ADNumberType>                        &scratch,
             PerTaskData_ASM                                      &data) const;
        void copy_local_to_global_system(const PerTaskData_ASM &data);

        // Define boundary conditions
        virtual void make_constraints(const int &it_nr);
        virtual void make_dirichlet_constraints(AffineConstraints<double> &constraints) = 0;
        virtual Tensor<1,dim> get_neumann_traction
             (const types::boundary_id &boundary_id,
              const Point<dim>         &pt,
              const Tensor<1,dim>      &N            ) const = 0;
        virtual double get_prescribed_fluid_flow
            (const types::boundary_id &boundary_id,
             const Point<dim>         &pt            ) const = 0;
        virtual std::pair<types::boundary_id,types::boundary_id>
                get_reaction_boundary_id_for_output () const = 0;
        virtual std::pair<types::boundary_id,types::boundary_id>
                get_drained_boundary_id_for_output () const = 0;
        virtual std::pair<double,FEValuesExtractors::Scalar>
                get_dirichlet_load(const types::boundary_id &boundary_id) const = 0;

        // Create and update the quadrature points.
        void setup_qph();

        //Solve non-linear system using a Newton-Raphson scheme
        void solve_nonlinear_timestep(TrilinosWrappers::MPI::BlockVector &solution_delta_OUT);

        //Solve the linearized equations using a direct solver
        void solve_linear_system (TrilinosWrappers::MPI::BlockVector &newton_update_OUT);

        //Retrieve the  solution
        TrilinosWrappers::MPI::BlockVector
          get_total_solution(const TrilinosWrappers::MPI::BlockVector &solution_delta_IN) const;

        // Store the converged values of the internal variables at the end of each timestep
        void update_end_timestep();

        //Post-processing and writing data to files
        void output_results_to_vtu(const unsigned int timestep,
                                   const double current_time,
                                   TrilinosWrappers::MPI::BlockVector solution) const;
        void output_bcs_to_vtu(const unsigned int timestep,
                               const double current_time,
                               TrilinosWrappers::MPI::BlockVector solution) const;
        void output_results_to_plot(const unsigned int timestep,
                                    const double current_time,
                                    TrilinosWrappers::MPI::BlockVector solution,
                                    std::vector<Point<dim> > &tracked_vertices,
                                    std::ofstream &pointfile) const;

        // Headers and footer for the output files
        void print_console_file_header( std::ofstream &outfile) const;
        void print_plot_file_header(std::vector<Point<dim> > &tracked_vertices,
                                    std::ofstream &pointfile) const;
        void print_console_file_footer(std::ofstream &outfile) const;
        void print_plot_file_footer(std::ofstream &pointfile) const;

        // For parallel communication
        MPI_Comm                   mpi_communicator;
        const unsigned int         n_mpi_processes;
        const unsigned int         this_mpi_process;
        mutable ConditionalOStream pcout;

        //A collection of the parameters used to describe the problem setup
        const Parameters::AllParameters &parameters;

        //Declare an instance of dealii Triangulation class (mesh)
        parallel::shared::Triangulation<dim>  triangulation;

        // Keep track of the current time and the time spent evaluating certain functions
        Time        time;
        TimerOutput timerconsole;
        TimerOutput timerfile;

        // A storage object for quadrature point information.
        CellDataStorage<typename Triangulation<dim>::cell_iterator, PointHistory<dim,ADNumberType> > quadrature_point_history;

        //Integers to store polynomial degree (needed for output)
        const unsigned int degree_displ;
        const unsigned int degree_pore;

        //Declare an instance of dealii FESystem class (finite element definition)
        const FESystem<dim> fe;

        //Declare an instance of dealii DoFHandler class (assign DoFs to mesh)
        DoFHandler<dim>    dof_handler_ref;

        //Integer to store DoFs per element (this value will be used often)
        const unsigned int dofs_per_cell;

        //Declare an instance of dealii Extractor objects used to retrieve
        //information from the solution vectors. We will use "u_fe" and
        //"p_fluid_fe"as subscript in operator [] expressions on FEValues
        //and FEFaceValues objects to extract the components of the
        //displacement vector and fluid pressure, respectively.
        const FEValuesExtractors::Vector u_fe;
        const FEValuesExtractors::Scalar ux_fe;
        const FEValuesExtractors::Scalar p_fluid_fe;

        // Description of how the block-system is arranged. There are 3 blocks:
        //  0 - vector DOF displacements u
        //  1 - scalar DOF fluid pressure p_fluid
        static const unsigned int n_blocks = 2;
        static const unsigned int n_components = dim+1;
        static const unsigned int first_u_component = 0;
        static const unsigned int p_fluid_component = dim;

        enum
        {
          u_block = 0,
          p_fluid_block = 1
        };

        // Extractors
        const FEValuesExtractors::Scalar x_displacement;
        const FEValuesExtractors::Scalar y_displacement;
        const FEValuesExtractors::Scalar z_displacement;
        const FEValuesExtractors::Scalar pressure;

        // Block data
        std::vector<unsigned int> block_component;

        // DoF index data
        std::vector<IndexSet> all_locally_owned_dofs;
        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;
        std::vector<IndexSet> locally_owned_partitioning;
        std::vector<IndexSet> locally_relevant_partitioning;

        std::vector<types::global_dof_index> dofs_per_block;
        std::vector<types::global_dof_index> element_indices_u;
        std::vector<types::global_dof_index> element_indices_p_fluid;

        //Declare an instance of dealii QGauss class (The Gauss-Legendre
        //family of quadrature rules for numerical integration)
        //Gauss Points in element, with n quadrature points
        //(in each space direction <dim>)
        const QGauss<dim>   qf_cell;
        //Gauss Points on element faces (used for definition of BCs)
        const QGauss<dim-1> qf_face;
        //Integer to store num GPs per element (this value will be used often)
        const unsigned int  n_q_points;
        //Integer to store num GPs per face (this value will be used often)
        const unsigned int  n_q_points_f;

        //Declare an instance of dealii AffineConstraints class (linear constraints on DoFs due to hanging nodes or BCs)
        AffineConstraints<double> constraints;

        //Declare an instance of dealii classes necessary for FE system set-up and assembly
        //Store elements of tangent matrix (indicated by SparsityPattern class) as sparse matrix (more efficient)
        TrilinosWrappers::BlockSparseMatrix tangent_matrix;
        TrilinosWrappers::BlockSparseMatrix tangent_matrix_preconditioner;
        //Right hand side vector of forces
        TrilinosWrappers::MPI::BlockVector  system_rhs;
        //Total displacement values + pressure (accumulated solution to FE system)
        TrilinosWrappers::MPI::BlockVector  solution_n;

        // Non-block system for the direct solver. We will copy the block system into these to solve the linearized system of equations.
        TrilinosWrappers::SparseMatrix tangent_matrix_nb;
        TrilinosWrappers::MPI::Vector  system_rhs_nb;

        //We define variables to store norms and update norms and normalisation factors.
        struct Errors
        {
          Errors()
            :
            norm(1.0), u(1.0), p_fluid(1.0)
          {}

          void reset()
          {
            norm = 1.0;
            u = 1.0;
            p_fluid = 1.0;
          }
          void normalise(const Errors &rhs)
          {
            if (rhs.norm != 0.0)
              norm /= rhs.norm;
            if (rhs.u != 0.0)
              u /= rhs.u;
            if (rhs.p_fluid != 0.0)
              p_fluid /= rhs.p_fluid;
          }

          double norm, u, p_fluid;
        };

        //Declare several instances of the "Error" structure
        Errors error_residual, error_residual_0, error_residual_norm, error_update,
               error_update_0, error_update_norm;

        // Methods to calculate error measures
        void get_error_residual(Errors &error_residual_OUT);
        void get_error_update
            (const TrilinosWrappers::MPI::BlockVector &newton_update_IN,
             Errors                                   &error_update_OUT);

        // Print information to screen
        void print_conv_header();
        void print_conv_footer();

//NOTE: In all functions, we pass by reference (&), so these functions work
//on the original copy (not a clone copy), modifying the input variables
//inside the functions will change them outside the function.
};

// @sect3{Implementation of the <code>Solid</code> class}
// @sect4{Public interface}
// We initialise the Solid class using data extracted from the parameter file.
template <int dim>
Solid<dim>::Solid(const Parameters::AllParameters &parameters)
    :
    mpi_communicator(MPI_COMM_WORLD),
    n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout(std::cout, this_mpi_process == 0),
    parameters(parameters),
    triangulation(mpi_communicator,Triangulation<dim>::maximum_smoothing),
    time(parameters.end_time, parameters.delta_t),
    timerconsole( mpi_communicator,
                  pcout,
                  TimerOutput::summary,
                  TimerOutput::wall_times),
    timerfile( mpi_communicator,
               outfile,
               TimerOutput::summary,
               TimerOutput::wall_times),
    degree_displ(parameters.poly_degree_displ),
    degree_pore(parameters.poly_degree_pore),
    fe( FE_Q<dim>(parameters.poly_degree_displ), dim,
        FE_Q<dim>(parameters.poly_degree_pore), 1 ),
    dof_handler_ref(triangulation),
    dofs_per_cell (fe.dofs_per_cell),
    u_fe(first_u_component),
    ux_fe(first_u_component),
    p_fluid_fe(p_fluid_component),
    x_displacement(first_u_component),
    y_displacement(first_u_component+1),
    z_displacement(first_u_component+2),
    pressure(p_fluid_component),
    dofs_per_block(n_blocks),
    qf_cell(parameters.quad_points),
    qf_face(parameters.quad_points),
    n_q_points (qf_cell.size()),
    n_q_points_f (qf_face.size())
    {
     Assert(dim==3, ExcMessage("This problem only works in 3 space dimensions."));
     determine_component_extractors();
    }

//The class destructor simply clears the data held by the DOFHandler
template <int dim>
Solid<dim>::~Solid()
{
    dof_handler_ref.clear();
}

//Runs the 3D solid problem
template <int dim>
void Solid<dim>::run()
{
      //The current solution increment is defined as a block vector to reflect the structure
      //of the PDE system, with multiple solution components
      TrilinosWrappers::MPI::BlockVector solution_delta;

      //Open file
      if (this_mpi_process == 0)
      {
          outfile.open("console-output.sol");
          print_console_file_header(outfile);
      }

      //Generate mesh
      make_grid();

      //Assign DOFs and create the stiffness and right-hand-side force vector
      system_setup(solution_delta);

      //Define points for post-processing
      std::vector<Point<dim>> tracked_vertices (2);
      define_tracked_vertices(tracked_vertices);
      std::vector<Point<dim>> reaction_force;

      if (this_mpi_process == 0)
      {
          pointfile.open("data-for-gnuplot.sol");
          print_plot_file_header(tracked_vertices, pointfile);
      }

      //Print results to output file
      if (parameters.outfiles_requested == "all")
      {
            output_results_to_vtu(time.get_timestep(),
                                  time.get_current(),
                                  solution_n           );
            output_bcs_to_vtu(time.get_timestep(),
                              time.get_current(),
                              solution_n           );
      }
      else if (parameters.outfiles_requested == "solution")
      {
            output_results_to_vtu(time.get_timestep(),
                                  time.get_current(),
                                  solution_n           );
      }
      else if (parameters.outfiles_requested == "bcs")
      {
            output_bcs_to_vtu(time.get_timestep(),
                              time.get_current(),
                              solution_n           );
      }

      output_results_to_plot(time.get_timestep(),
                             time.get_current(),
                             solution_n,
                             tracked_vertices,
                             pointfile);

      //Increment time step (=load step)
      //NOTE: In solving the quasi-static problem, the time becomes a loading
      //parameter, i.e. we increase the loading linearly with time, making
      //the two concepts interchangeable.
      time.increment_time();

      //Print information on screen
      pcout << "\nSolver:";
      pcout << "\n  CST     = make constraints";
      pcout << "\n  ASM_SYS = assemble system";
      pcout << "\n  SLV     = linear solver \n";

      //Print information on file
      outfile << "\nSolver:";
      outfile << "\n  CST     = make constraints";
      outfile << "\n  ASM_SYS = assemble system";
      outfile << "\n  SLV     = linear solver \n";

      while ( (time.get_end()-time.get_current())>-1.0*parameters.tol_u )
        {
          //Initialize the current solution increment to zero
          solution_delta = 0.0;

          //Solve the non-linear system using a Newton-Rapshon scheme
          solve_nonlinear_timestep(solution_delta);

          //Add the computed solution increment to total solution
          solution_n += solution_delta;

          //Store the converged values of the internal variables
          update_end_timestep();

          //Output results
          if ( (time.get_timestep()%parameters.timestep_output) == 0 )
          {
            if (parameters.outfiles_requested == "all")
            {
                  output_results_to_vtu(time.get_timestep(),
                                        time.get_current(),
                                        solution_n           );
                  output_bcs_to_vtu(time.get_timestep(),
                                    time.get_current(),
                                    solution_n           );
            }
            else if (parameters.outfiles_requested == "solution")
            {
                  output_results_to_vtu(time.get_timestep(),
                                        time.get_current(),
                                        solution_n           );
            }
            else if (parameters.outfiles_requested == "bcs")
            {
                  output_bcs_to_vtu(time.get_timestep(),
                                    time.get_current(),
                                    solution_n           );
            }
          }

          output_results_to_plot(time.get_timestep(),
                                 time.get_current(),
                                 solution_n,
                                 tracked_vertices,
                                 pointfile);

          //Increment the time step (=load step)
          time.increment_time();
        }

      //Print the footers and close files
      if (this_mpi_process == 0)
      {
          print_plot_file_footer(pointfile);
          pointfile.close ();
          print_console_file_footer(outfile);
      //NOTE: ideally, we should close the outfile here [>>outfile.close();]
      //But if we do, then the timer output will not be printed.
      //That is why we leave it open.
      }
}

// @sect4{Private interface}
// We define the structures needed for parallelization with Threading
// Building Blocks (TBB). Tangent matrix and right-hand side force vector
// assembly structures. PerTaskData_ASM stores local contributions
template <int dim>
struct Solid<dim>::PerTaskData_ASM
{
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    PerTaskData_ASM(const unsigned int dofs_per_cell)
      :
      cell_matrix(dofs_per_cell, dofs_per_cell),
      cell_rhs(dofs_per_cell),
      local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      cell_matrix = 0.0;
      cell_rhs = 0.0;
    }
};

// ScratchData_ASM stores larger objects used during the assembly
template <int dim>
template <typename NumberType>
struct Solid<dim>::ScratchData_ASM
{
    const TrilinosWrappers::MPI::BlockVector &solution_total;

    //Integration helper
    FEValues<dim>     fe_values_ref;
    FEFaceValues<dim> fe_face_values_ref;

    // Quadrature point solution
    std::vector<NumberType>               local_dof_values;
    std::vector<Tensor<2,dim,NumberType>> solution_grads_u_total;
    std::vector<Tensor<3,dim,NumberType>> solution_hess_u_total;
    std::vector<NumberType>               solution_values_p_fluid_total;
    std::vector<Tensor<1,dim,NumberType>> solution_grads_p_fluid_total;
    std::vector<Tensor<2,dim,NumberType>> solution_hess_p_fluid_total;
    std::vector<Tensor<1,dim,NumberType>> solution_grads_face_p_fluid_total;

    //shape function values
    std::vector<std::vector<Tensor<1,dim>>> Nx;
    std::vector<std::vector<double>>        Nx_p_fluid;
    //shape function gradients
    std::vector<std::vector<Tensor<2,dim,NumberType>>>          grad_Nx;
    std::vector<std::vector<SymmetricTensor<2,dim,NumberType>>> symm_grad_Nx;
    std::vector<std::vector<Tensor<1,dim,NumberType>>>          grad_Nx_p_fluid;

    ScratchData_ASM(const FiniteElement<dim> &fe_cell,
                    const QGauss<dim>   &qf_cell, const UpdateFlags uf_cell,
                    const QGauss<dim-1> & qf_face, const UpdateFlags uf_face,
                    const TrilinosWrappers::MPI::BlockVector &solution_total )
      :
      solution_total (solution_total),
      fe_values_ref(fe_cell, qf_cell, uf_cell),
      fe_face_values_ref(fe_cell, qf_face, uf_face),
      local_dof_values(fe_cell.dofs_per_cell),
      solution_grads_u_total(qf_cell.size()),
      solution_hess_u_total(qf_cell.size()),
      solution_values_p_fluid_total(qf_cell.size()),
      solution_grads_p_fluid_total(qf_cell.size()),
      solution_hess_p_fluid_total(qf_cell.size()),
      solution_grads_face_p_fluid_total(qf_face.size()),
      Nx(qf_cell.size(), std::vector<Tensor<1,dim>>(fe_cell.dofs_per_cell)),
      Nx_p_fluid(qf_cell.size(), std::vector<double>(fe_cell.dofs_per_cell)),
      grad_Nx(qf_cell.size(),
              std::vector<Tensor<2,dim,NumberType>>(fe_cell.dofs_per_cell)),
      symm_grad_Nx(qf_cell.size(),
                   std::vector<SymmetricTensor<2,dim,NumberType>>(fe_cell.dofs_per_cell)),
      grad_Nx_p_fluid(qf_cell.size(),
                      std::vector<Tensor<1,dim,NumberType>>(fe_cell.dofs_per_cell))
    {}

    ScratchData_ASM(const ScratchData_ASM &rhs)
      :
      solution_total (rhs.solution_total),
      fe_values_ref(rhs.fe_values_ref.get_fe(),
                    rhs.fe_values_ref.get_quadrature(),
                    rhs.fe_values_ref.get_update_flags()),
      fe_face_values_ref(rhs.fe_face_values_ref.get_fe(),
                         rhs.fe_face_values_ref.get_quadrature(),
                         rhs.fe_face_values_ref.get_update_flags()),
      local_dof_values(rhs.local_dof_values),
      solution_grads_u_total(rhs.solution_grads_u_total),
      solution_hess_u_total(rhs.solution_hess_u_total),
      solution_values_p_fluid_total(rhs.solution_values_p_fluid_total),
      solution_grads_p_fluid_total(rhs.solution_grads_p_fluid_total),
      solution_hess_p_fluid_total(rhs.solution_hess_p_fluid_total),
      solution_grads_face_p_fluid_total(rhs.solution_grads_face_p_fluid_total),
      Nx(rhs.Nx),
      Nx_p_fluid(rhs.Nx_p_fluid),
      grad_Nx(rhs.grad_Nx),
      symm_grad_Nx(rhs.symm_grad_Nx),
      grad_Nx_p_fluid(rhs.grad_Nx_p_fluid)
    {}

    void reset()
    {
      const unsigned int n_q_points = Nx_p_fluid.size();
      const unsigned int n_dofs_per_cell = Nx_p_fluid[0].size();

      Assert(local_dof_values.size() == n_dofs_per_cell, ExcInternalError());

      for (unsigned int k=0; k<n_dofs_per_cell; ++k)
        {
          local_dof_values[k] = 0.0;
        }

      Assert(solution_grads_u_total.size() == n_q_points, ExcInternalError());
      Assert(solution_hess_u_total.size() == n_q_points, ExcInternalError());
      Assert(solution_values_p_fluid_total.size() == n_q_points, ExcInternalError());
      Assert(solution_grads_p_fluid_total.size() == n_q_points, ExcInternalError());
      Assert(solution_hess_p_fluid_total.size() == n_q_points, ExcInternalError());

      Assert(Nx.size() == n_q_points, ExcInternalError());
      Assert(grad_Nx.size() == n_q_points, ExcInternalError());
      Assert(symm_grad_Nx.size() == n_q_points, ExcInternalError());

      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
          Assert(Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
          Assert(grad_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
          Assert(symm_grad_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());

          solution_grads_u_total[q_point] = 0.0;
          solution_hess_u_total[q_point] = 0.0;
          solution_values_p_fluid_total[q_point] = 0.0;
          solution_grads_p_fluid_total[q_point] = 0.0;
          solution_hess_p_fluid_total[q_point] = 0.0;

          for (unsigned int k=0; k<n_dofs_per_cell; ++k)
            {
              Nx[q_point][k] = 0.0;
              Nx_p_fluid[q_point][k] = 0.0;
              grad_Nx[q_point][k] = 0.0;
              symm_grad_Nx[q_point][k] = 0.0;
              grad_Nx_p_fluid[q_point][k] = 0.0;
            }
        }

      const unsigned int n_f_q_points = solution_grads_face_p_fluid_total.size();
      Assert(solution_grads_face_p_fluid_total.size() == n_f_q_points, ExcInternalError());

      for (unsigned int f_q_point=0; f_q_point<n_f_q_points; ++f_q_point)
          solution_grads_face_p_fluid_total[f_q_point] = 0.0;
    }
};

//Define the boundary conditions on the mesh
template <int dim>
void Solid<dim>::make_constraints(const int &it_nr_IN)
{
    pcout     << " CST " << std::flush;
    outfile   << " CST " << std::flush;

    if (it_nr_IN > 1) return;

    const bool apply_dirichlet_bc = (it_nr_IN == 0);

    if (apply_dirichlet_bc)
    {
      constraints.clear();
      make_dirichlet_constraints(constraints);
    }
    else
    {
      for (unsigned int i=0; i<dof_handler_ref.n_dofs(); ++i)
        if (constraints.is_inhomogeneously_constrained(i) == true)
          constraints.set_inhomogeneity(i,0.0);
    }
    constraints.close();
}

//Set-up the FE system
template <int dim>
void Solid<dim>::system_setup(TrilinosWrappers::MPI::BlockVector &solution_delta_OUT)
{
    timerconsole.enter_subsection("Setup system");
    timerfile.enter_subsection("Setup system");


    //Determine number of components per block
    std::vector<unsigned int> block_component(n_components, u_block);
    block_component[p_fluid_component] = p_fluid_block;

    // The DOF handler is initialised and we renumber the grid in an efficient manner.
    dof_handler_ref.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler_ref);
    DoFRenumbering::component_wise(dof_handler_ref, block_component);

    // Count the number of DoFs in each block
    dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler_ref, block_component);

    // Setup the sparsity pattern and tangent matrix
    all_locally_owned_dofs = DoFTools::locally_owned_dofs_per_subdomain(dof_handler_ref);
    std::vector<IndexSet> all_locally_relevant_dofs
    = DoFTools::locally_relevant_dofs_per_subdomain(dof_handler_ref);

    locally_owned_dofs.clear();
    locally_owned_partitioning.clear();
    Assert(all_locally_owned_dofs.size() > this_mpi_process, ExcInternalError());
    locally_owned_dofs = all_locally_owned_dofs[this_mpi_process];

    locally_relevant_dofs.clear();
    locally_relevant_partitioning.clear();
    Assert(all_locally_relevant_dofs.size() > this_mpi_process, ExcInternalError());
    locally_relevant_dofs = all_locally_relevant_dofs[this_mpi_process];

    locally_owned_partitioning.reserve(n_blocks);
    locally_relevant_partitioning.reserve(n_blocks);

    for (unsigned int b=0; b<n_blocks; ++b)
      {
        const types::global_dof_index idx_begin
        = std::accumulate(dofs_per_block.begin(),
                          std::next(dofs_per_block.begin(),b), 0);
        const types::global_dof_index idx_end
        = std::accumulate(dofs_per_block.begin(),
                          std::next(dofs_per_block.begin(),b+1), 0);
        locally_owned_partitioning.push_back(
                          locally_owned_dofs.get_view(idx_begin, idx_end));
        locally_relevant_partitioning.push_back(
                          locally_relevant_dofs.get_view(idx_begin, idx_end));
      }

    //Print information on screen
    pcout << "\nTriangulation:\n"
          << "  Number of active cells: "
          << triangulation.n_active_cells()
          << " (by partition:";
    for (unsigned int p=0; p<n_mpi_processes; ++p)
      pcout << (p==0 ? ' ' : '+')
            << (GridTools::count_cells_with_subdomain_association (triangulation,p));
    pcout << ")"
          << std::endl;
    pcout << "  Number of degrees of freedom: "
          << dof_handler_ref.n_dofs()
          << " (by partition:";
    for (unsigned int p=0; p<n_mpi_processes; ++p)
      pcout << (p==0 ? ' ' : '+')
            << (DoFTools::count_dofs_with_subdomain_association (dof_handler_ref,p));
    pcout << ")"
          << std::endl;
    pcout << "  Number of degrees of freedom per block: "
          << "[n_u, n_p_fluid] = ["
          << dofs_per_block[u_block]
          << ", "
          << dofs_per_block[p_fluid_block]
          << "]"
          << std::endl;

    //Print information to file
    outfile << "\nTriangulation:\n"
            <<  "  Number of active cells: "
            << triangulation.n_active_cells()
            << " (by partition:";
    for (unsigned int p=0; p<n_mpi_processes; ++p)
      outfile << (p==0 ? ' ' : '+')
              << (GridTools::count_cells_with_subdomain_association(triangulation,p));
    outfile << ")"
            << std::endl;
    outfile << "  Number of degrees of freedom: "
            << dof_handler_ref.n_dofs()
            << " (by partition:";
    for (unsigned int p=0; p<n_mpi_processes; ++p)
      outfile << (p==0 ? ' ' : '+')
              << (DoFTools::count_dofs_with_subdomain_association(dof_handler_ref,p));
    outfile << ")"
            << std::endl;
    outfile << "  Number of degrees of freedom per block: "
            << "[n_u, n_p_fluid] = ["
            << dofs_per_block[u_block]
            << ", "
            << dofs_per_block[p_fluid_block]
            << "]"
            << std::endl;

    // We optimise the sparsity pattern to reflect this structure and prevent
    // unnecessary data creation for the right-diagonal block components.
    Table<2,DoFTools::Coupling>coupling(n_components, n_components);
    for (unsigned int ii=0; ii<n_components; ++ii)
      for (unsigned int jj=0; jj<n_components; ++jj)

        //Identify "zero" matrix components of FE-system
        //(The two components do not couple)
        if (((ii==p_fluid_component) && (jj<p_fluid_component))
            || ((ii<p_fluid_component) && (jj==p_fluid_component)) )
          coupling[ii][jj] = DoFTools::none;

        //The rest of components always couple
        else
          coupling[ii][jj] = DoFTools::always;

    TrilinosWrappers::BlockSparsityPattern bsp (locally_owned_partitioning,
                                                mpi_communicator);

    DoFTools::make_sparsity_pattern (dof_handler_ref, bsp, constraints,
                                     false, this_mpi_process);
    bsp.compress();

    //Reinitialize the (sparse) tangent matrix with the given sparsity pattern.
    tangent_matrix.reinit (bsp);

    //Initialize the right hand side and solution vectors with number of DoFs
    system_rhs.reinit(locally_owned_partitioning, mpi_communicator);
    solution_n.reinit(locally_owned_partitioning, mpi_communicator);
    solution_delta_OUT.reinit(locally_owned_partitioning, mpi_communicator);

    // Non-block system
    TrilinosWrappers::SparsityPattern sp (locally_owned_dofs,
                                          mpi_communicator);
    DoFTools::make_sparsity_pattern (dof_handler_ref, sp, constraints,
                                     false, this_mpi_process);
    sp.compress();
    tangent_matrix_nb.reinit (sp);
    system_rhs_nb.reinit(locally_owned_dofs, mpi_communicator);

    //Set up the quadrature point history
    setup_qph();

    timerconsole.leave_subsection();
    timerfile.leave_subsection();
}

//Component extractors: used to extract sub-blocks from the global matrix
//Description of which local element DOFs are attached to which block component
template <int dim>
void Solid<dim>::determine_component_extractors()
{
    element_indices_u.clear();
    element_indices_p_fluid.clear();

    for (unsigned int k=0; k<fe.dofs_per_cell; ++k)
    {
        const unsigned int k_group = fe.system_to_base_index(k).first.first;
        if (k_group == u_block)
          element_indices_u.push_back(k);
        else if (k_group == p_fluid_block)
          element_indices_p_fluid.push_back(k);
        else
            Assert(k_group <= p_fluid_block, ExcInternalError());
    }
}

//Set-up quadrature point history (QPH) data objects
template <int dim>
void Solid<dim>::setup_qph()
{
    pcout     << "\nSetting up quadrature point data..." << std::endl;
    outfile   << "\nSetting up quadrature point data..." << std::endl;

    //Create QPH data objects.
    quadrature_point_history.initialize(triangulation.begin_active(),
                                        triangulation.end(), n_q_points);

    //Setup the initial quadrature point data using the info stored in parameters
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    cell (IteratorFilters::LocallyOwnedCell(),
          dof_handler_ref.begin_active()),
    endc (IteratorFilters::LocallyOwnedCell(),
          dof_handler_ref.end());
    for (; cell!=endc; ++cell)
      {
        Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
        const std::vector<std::shared_ptr<PointHistory<dim,ADNumberType>>>
            lqph = quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            lqph[q_point]->setup_lqp(parameters, time);
      }
}

//Solve the non-linear system using a Newton-Raphson scheme
template <int dim>
void Solid<dim>::solve_nonlinear_timestep
 (TrilinosWrappers::MPI::BlockVector &solution_delta_OUT)
{
    //Print the load step
    pcout << std::endl << "\nTimestep "
          << time.get_timestep()
          << " @ "
          << time.get_current()
          << "s" << std::endl;
    outfile << std::endl << "\nTimestep "
            << time.get_timestep()
            << " @ "
            << time.get_current()
            << "s" << std::endl;

    //Declare newton_update vector (solution of a Newton iteration),
    //which must have as many positions as global DoFs.
    TrilinosWrappers::MPI::BlockVector newton_update
      (locally_owned_partitioning, mpi_communicator);

    //Reset the error storage objects
    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();

    print_conv_header();

    //Declare and initialize iterator for the Newton-Raphson algorithm steps
    unsigned int newton_iteration = 0;

    //Iterate until error is below tolerance or max number iterations are reached
    while(newton_iteration<parameters.max_iterations_NR)
    {
        pcout   <<" "<< std::setw(2) << newton_iteration <<" "<< std::flush;
        outfile <<" "<< std::setw(2) << newton_iteration <<" "<< std::flush;

        //Initialize global stiffness matrix and global force vector to zero
        tangent_matrix = 0.0;
        system_rhs = 0.0;

        tangent_matrix_nb = 0.0;
        system_rhs_nb = 0.0;

        //Apply boundary conditions
        make_constraints(newton_iteration);
        assemble_system(solution_delta_OUT);

        //Compute the rhs residual (error between external and internal forces
        //in FE system)
        get_error_residual(error_residual);

        //error_residual in first iteration is stored to normalize posterior
        //error measures
        if (newton_iteration == 0)
          error_residual_0 = error_residual;

        // Determine the normalised residual error
        error_residual_norm = error_residual;
        error_residual_norm.normalise(error_residual_0);

        //If both errors are below the tolerances, exit the loop.
        // We need to check the residual vector directly for convergence
        // in the load steps where no external forces or displacements are imposed.
        if (  ((newton_iteration > 0)
            && (error_update_norm.u <= parameters.tol_u)
            && (error_update_norm.p_fluid <= parameters.tol_p_fluid)
            && (error_residual_norm.u <= parameters.tol_f)
            && (error_residual_norm.p_fluid  <= parameters.tol_f))
            || ( (newton_iteration > 0)
                && system_rhs.l2_norm() <= parameters.tol_f) )
          {
            pcout   << "\n ***** CONVERGED! *****     "
                    << system_rhs.l2_norm() << "      "
                    << "  " << error_residual_norm.norm
                    << "  " << error_residual_norm.u
                    << "  " << error_residual_norm.p_fluid
                    << "        " << error_update_norm.norm
                    << "  " << error_update_norm.u
                    << "  " << error_update_norm.p_fluid
                    << "  " << std::endl;
            outfile   << "\n ***** CONVERGED! *****     "
                    << system_rhs.l2_norm() << "      "
                    << "  " << error_residual_norm.norm
                    << "  " << error_residual_norm.u
                    << "  " << error_residual_norm.p_fluid
                    << "        " << error_update_norm.norm
                    << "  " << error_update_norm.u
                    << "  " << error_update_norm.p_fluid
                    << "  " << std::endl;
            print_conv_footer();

            break;
        }

        //Solve the linearized system
        solve_linear_system(newton_update);
        constraints.distribute(newton_update);

        //Compute the displacement error
        get_error_update(newton_update, error_update);

        //error_update in the first iteration is stored to normalize
        //posterior error measures
        if (newton_iteration == 0)
          error_update_0 = error_update;

        // Determine the normalised Newton update error
        error_update_norm = error_update;
        error_update_norm.normalise(error_update_0);

        // Determine the normalised residual error
        error_residual_norm = error_residual;
        error_residual_norm.normalise(error_residual_0);

        //Print error values
        pcout    << " |   " << std::fixed << std::setprecision(3)
        << std::setw(7) << std::scientific
        << system_rhs.l2_norm()
        << "        " << error_residual_norm.norm
        << "  " << error_residual_norm.u
        << "  " << error_residual_norm.p_fluid
        << "        " << error_update_norm.norm
        << "  " << error_update_norm.u
        << "  " << error_update_norm.p_fluid
        << "  " << std::endl;

        outfile  << " |   " << std::fixed << std::setprecision(3)
        << std::setw(7) << std::scientific
        << system_rhs.l2_norm()
        << "        " << error_residual_norm.norm
        << "  " << error_residual_norm.u
        << "  " << error_residual_norm.p_fluid
        << "        " << error_update_norm.norm
        << "  " << error_update_norm.u
        << "  " << error_update_norm.p_fluid
        << "  " << std::endl;

        // Update
        solution_delta_OUT += newton_update;
        newton_update = 0.0;
        newton_iteration++;
      }

    //If maximum allowed number of iterations for Newton algorithm are reached,
    //print non-convergence message and abort program
    AssertThrow (newton_iteration<parameters.max_iterations_NR,
                 ExcMessage("No convergence in nonlinear solver!"));
}

//Prints the header for convergence info on console
template <int dim>
void Solid<dim>::print_conv_header()
{
    static const unsigned int l_width = 120;

    for (unsigned int i=0; i<l_width; ++i)
    {
        pcout     << "_";
        outfile   << "_";
    }

    pcout   << std::endl;
    outfile << std::endl;

    pcout   << "\n       SOLVER STEP      |    SYS_RES         "
            << "RES_NORM     RES_U      RES_P           "
            << "NU_NORM     NU_U       NU_P " << std::endl;
    outfile << "\n       SOLVER STEP      |    SYS_RES         "
            << "RES_NORM     RES_U      RES_P           "
            << "NU_NORM     NU_U       NU_P " << std::endl;

    for (unsigned int i=0; i<l_width; ++i)
    {
        pcout     << "_";
        outfile   << "_";
    }
    pcout   << std::endl << std::endl;
    outfile << std::endl << std::endl;
}

//Prints the footer for convergence info on console
template <int dim>
void Solid<dim>::print_conv_footer()
{
    static const unsigned int l_width = 120;

    for (unsigned int i = 0; i < l_width; ++i)
    {
        pcout     << "_";
        outfile   << "_";
    }
    pcout   << std::endl << std::endl;
    outfile << std::endl << std::endl;

    pcout << "Relative errors:" << std::endl
          << "Displacement:  "
          << error_update.u / error_update_0.u  << std::endl
          << "Force (displ): "
          << error_residual.u / error_residual_0.u << std::endl
          << "Pore pressure: "
          << error_update.p_fluid / error_update_0.p_fluid << std::endl
          << "Force (pore):  "
          << error_residual.p_fluid / error_residual_0.p_fluid << std::endl;
    outfile << "Relative errors:" << std::endl
            << "Displacement:  "
            << error_update.u / error_update_0.u << std::endl
            << "Force (displ): "
            << error_residual.u / error_residual_0.u << std::endl
            << "Pore pressure: "
            << error_update.p_fluid / error_update_0.p_fluid << std::endl
            << "Force (pore):  "
            << error_residual.p_fluid / error_residual_0.p_fluid << std::endl;
}

//Determine the true residual error for the problem
template <int dim>
void Solid<dim>::get_error_residual(Errors &error_residual_OUT)
{
    TrilinosWrappers::MPI::BlockVector error_res(system_rhs);
    constraints.set_zero(error_res);

    error_residual_OUT.norm = error_res.l2_norm();
    error_residual_OUT.u = error_res.block(u_block).l2_norm();
    error_residual_OUT.p_fluid = error_res.block(p_fluid_block).l2_norm();
}

//Determine the true Newton update error for the problem
template <int dim>
void Solid<dim>::get_error_update
                  (const TrilinosWrappers::MPI::BlockVector &newton_update_IN,
                   Errors                                   &error_update_OUT)
{
    TrilinosWrappers::MPI::BlockVector error_ud(newton_update_IN);
    constraints.set_zero(error_ud);

    error_update_OUT.norm = error_ud.l2_norm();
    error_update_OUT.u = error_ud.block(u_block).l2_norm();
    error_update_OUT.p_fluid = error_ud.block(p_fluid_block).l2_norm();
}

//Compute the total solution, which is valid at any Newton step. This is
//required as, to reduce computational error, the total solution is only
//updated at the end of the timestep.
template <int dim>
TrilinosWrappers::MPI::BlockVector
Solid<dim>::get_total_solution
          (const TrilinosWrappers::MPI::BlockVector &solution_delta_IN) const
{
    // Cell interpolation -> Ghosted vector
    TrilinosWrappers::MPI::BlockVector
        solution_total(locally_owned_partitioning,
                       locally_relevant_partitioning,
                       mpi_communicator,
                       /*vector_writable = */ false);
    TrilinosWrappers::MPI::BlockVector tmp (solution_total);
    solution_total = solution_n;
    tmp = solution_delta_IN;
    solution_total += tmp;
    return solution_total;
}

//Compute elemental stiffness tensor and right-hand side force vector,
//and assemble into global ones
template <int dim>
void Solid<dim>::assemble_system
  (const TrilinosWrappers::MPI::BlockVector &solution_delta)
{
    timerconsole.enter_subsection("Assemble system");
    timerfile.enter_subsection("Assemble system");
    pcout   << " ASM_SYS " << std::flush;
    outfile << " ASM_SYS " << std::flush;

    const TrilinosWrappers::MPI::BlockVector
        solution_total(get_total_solution(solution_delta));

    //Info given to FEValues and FEFaceValues constructors, to indicate
    //which data will be needed at each element.
    const UpdateFlags uf_cell(update_values |
                              update_gradients | update_hessians |
                              update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_gradients |
                              update_normal_vectors |
                              update_quadrature_points | update_JxW_values);

    //Setup a copy of the data structures required for the process and pass them, along with the
    //memory addresses of the assembly functions to the WorkStream object for processing
    PerTaskData_ASM per_task_data(dofs_per_cell);
    ScratchData_ASM<ADNumberType> scratch_data(fe, qf_cell, uf_cell,
                                               qf_face, uf_face,
                                               solution_total);

    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    cell (IteratorFilters::LocallyOwnedCell(),
          dof_handler_ref.begin_active()),
    endc (IteratorFilters::LocallyOwnedCell(),
          dof_handler_ref.end());
    for (; cell != endc; ++cell)
      {
        Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
        assemble_system_one_cell(cell, scratch_data, per_task_data);
        copy_local_to_global_system(per_task_data);
      }
    tangent_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    tangent_matrix_nb.compress(VectorOperation::add);
    system_rhs_nb.compress(VectorOperation::add);

    timerconsole.leave_subsection();
    timerfile.leave_subsection();
}

//Add the local elemental contribution to the global stiffness tensor
// We do it twice, for the block and the non-block systems
template <int dim>
void Solid<dim>::copy_local_to_global_system (const PerTaskData_ASM &data)
{
    constraints.distribute_local_to_global(data.cell_matrix,
        data.cell_rhs,
        data.local_dof_indices,
        tangent_matrix,
        system_rhs);

    constraints.distribute_local_to_global(data.cell_matrix,
        data.cell_rhs,
        data.local_dof_indices,
        tangent_matrix_nb,
        system_rhs_nb);
}

//Compute stiffness matrix and corresponding rhs for one element
template <int dim>
void Solid<dim>::assemble_system_one_cell
      (const typename DoFHandler<dim>::active_cell_iterator &cell,
       ScratchData_ASM<ADNumberType>                        &scratch,
       PerTaskData_ASM                                      &data) const
{
    Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());

    data.reset();
    scratch.reset();
    scratch.fe_values_ref.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    // Setup automatic differentiation
    for (unsigned int k=0; k<dofs_per_cell; ++k)
      {
        // Initialise the dofs for the cell using the current solution.
        scratch.local_dof_values[k] = scratch.solution_total[data.local_dof_indices[k]];
        // Mark this cell DoF as an independent variable
        scratch.local_dof_values[k].diff(k, dofs_per_cell);
      }

    // Update the quadrature point solution
    // Compute the values and gradients of the solution in terms of the AD variables
    for (unsigned int q=0; q<n_q_points; ++q)
    {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
            const unsigned int k_group = fe.system_to_base_index(k).first.first;
            if (k_group == u_block)
            {
                const Tensor<2, dim> Grad_Nx_u =
                                  scratch.fe_values_ref[u_fe].gradient(k, q);
                const Tensor<3, dim> Hessian_Nx_u =
                                  scratch.fe_values_ref[u_fe].hessian(k, q);

                for (unsigned int dd=0; dd<dim; dd++)
                    for (unsigned int ee=0; ee<dim; ee++)
                    {
                        // Gradient of displ
                        scratch.solution_grads_u_total[q][dd][ee]
                          += scratch.local_dof_values[k] * Grad_Nx_u[dd][ee];

                        for (unsigned int ff=0; ff<dim; ff++)
                          // Hessian of displ
                          scratch.solution_hess_u_total[q][dd][ee][ff]
                            += scratch.local_dof_values[k] * Hessian_Nx_u[dd][ee][ff];
                    }
            }
            else if  (k_group == p_fluid_block)
            {
                const double Nx_p = scratch.fe_values_ref[p_fluid_fe].value(k, q);
                const Tensor<1, dim> Grad_Nx_p =
                            scratch.fe_values_ref[p_fluid_fe].gradient(k, q);
                const Tensor<2, dim> Hessian_Nx_p =
                            scratch.fe_values_ref[p_fluid_fe].hessian(k, q);

                // Value of pressure
                scratch.solution_values_p_fluid_total[q]
                          += scratch.local_dof_values[k] * Nx_p;

                for (unsigned int dd = 0; dd < dim; dd++)
                {
                    // Gradient of pressure
                    scratch.solution_grads_p_fluid_total[q][dd]
                      += scratch.local_dof_values[k] * Grad_Nx_p[dd];
                    for (unsigned int ee=0; ee<dim; ee++)
                    // Hessian of pressure
                    scratch.solution_hess_p_fluid_total[q][dd][ee]
                      += scratch.local_dof_values[k] * Hessian_Nx_p[dd][ee];

                }
            }
            else
              Assert(k_group <= p_fluid_block, ExcInternalError());

        }
    }

    //Set up pointer "lgph" to the PointHistory object of this element
    const std::vector<std::shared_ptr<const PointHistory<dim,ADNumberType>>>
        lqph = quadrature_point_history.get_data(cell);
    Assert(lqph.size() == n_q_points, ExcInternalError());


    //Precalculate the element shape function values and gradients
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {
        Tensor<2,dim,ADNumberType> F_AD = scratch.solution_grads_u_total[q_point];
        F_AD += Tensor<2,dim,double>(Physics::Elasticity::StandardTensors<dim>::I);
        Assert(determinant(F_AD)>0, ExcMessage("Invalid deformation map"));
        const Tensor<2,dim,ADNumberType> F_inv_AD = invert(F_AD);

        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            const unsigned int i_group = fe.system_to_base_index(i).first.first;

            if (i_group == u_block)
            {
                scratch.Nx[q_point][i] =
                    scratch.fe_values_ref[u_fe].value(i, q_point);
                scratch.grad_Nx[q_point][i] =
                    scratch.fe_values_ref[u_fe].gradient(i, q_point)*F_inv_AD;
                scratch.symm_grad_Nx[q_point][i] =
                    symmetrize(scratch.grad_Nx[q_point][i]);
            }
            else if  (i_group == p_fluid_block)
            {
                scratch.Nx_p_fluid[q_point][i] =
                    scratch.fe_values_ref[p_fluid_fe].value(i, q_point);
                scratch.grad_Nx_p_fluid[q_point][i] =
                    scratch.fe_values_ref[p_fluid_fe].gradient(i, q_point)*F_inv_AD;
            }
            else
              Assert(i_group <= p_fluid_block, ExcInternalError());
        }
      }

    //Assemble the stiffness matrix and rhs vector
    std::vector<ADNumberType> residual_ad (dofs_per_cell, ADNumberType(0.0));
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {
        // Deformation gradient tensor
        Tensor<2,dim,ADNumberType> F_AD = scratch.solution_grads_u_total[q_point];
        F_AD += Tensor<2,dim,double>(Physics::Elasticity::StandardTensors<dim>::I);
        const ADNumberType det_F_AD = determinant(F_AD);
        Assert(det_F_AD>0, ExcInternalError());
        const Tensor<2,dim,ADNumberType> F_inv_AD = invert(F_AD);
        const Tensor<2,dim,ADNumberType> F_inv_T_AD = transpose(F_inv_AD);


        // Define some aliases to make the assembly process easier to follow
        const std::vector<Tensor<1,dim>> &Nu = scratch.Nx[q_point];
        const std::vector<SymmetricTensor<2,dim,ADNumberType>> &symm_grad_Nu
                                          = scratch.symm_grad_Nx[q_point];
        const std::vector<double> &Np = scratch.Nx_p_fluid[q_point];
        const std::vector<Tensor<1,dim,ADNumberType>> &grad_Np
                                        = scratch.grad_Nx_p_fluid[q_point];

        // Pressure
        const ADNumberType p_fluid_AD =
                                scratch.solution_values_p_fluid_total[q_point];
        // Gradient of pressure
        const Tensor<1,dim,ADNumberType> grad_p_fluid_AD
                  = scratch.solution_grads_p_fluid_total[q_point]*F_inv_AD;
        // Hessian of pressure
        const Tensor<2,dim,ADNumberType> hess_p_fluid_AD
              = F_inv_T_AD*(scratch.solution_hess_p_fluid_total[q_point])*F_inv_AD;

       // Hessian of displacements in ref configuration
       const Tensor<3,dim,ADNumberType> hess_u_AD
            = scratch.solution_hess_u_total[q_point];

        // Gradient of determinat of F
        const Tensor<1,dim,ADNumberType> grad_det_F_AD = det_F_AD * double_contract<0,0,1,1>(transpose(F_inv_AD), hess_u_AD);

        // Quadrature weight
        const double JxW = scratch.fe_values_ref.JxW(q_point);

        // Current quadrature point
        const Point<dim> pt = scratch.fe_values_ref.quadrature_point(q_point);

        // Update internal equilibrium
        {
          PointHistory<dim, ADNumberType> *lqph_q_point_nc
              = const_cast<PointHistory<dim,ADNumberType>*>(lqph[q_point].get());

          const ADNumberType div_seepage_vel_AD
              =  lqph_q_point_nc->get_div_seepage_vel(grad_p_fluid_AD,
                                                      hess_p_fluid_AD,
                                                      grad_det_F_AD,
                                                      F_AD);

          lqph_q_point_nc->update_internal_equilibrium(F_AD, p_fluid_AD, div_seepage_vel_AD, pt);
        }

        //Growth
        const Tensor<2,dim> Fg = lqph[q_point]->get_non_converged_growth_tensor();
        const Tensor<2,dim> Fg_inv = invert(Fg);
        const Tensor<2,dim,ADNumberType> Fve_AD = F_AD * Fg_inv;
        const ADNumberType det_Fve_AD = determinant(Fve_AD);
        Assert(det_Fve_AD>0, ExcInternalError());

        //Get some info from constitutive model of solid
        static const SymmetricTensor<2,dim,double>
            I(Physics::Elasticity::StandardTensors<dim>::I);
        const SymmetricTensor<2,dim,ADNumberType> tau_E
                                          = lqph[q_point]->get_tau_E(F_AD);
        SymmetricTensor<2,dim,ADNumberType> tau_fluid_vol(I);
        tau_fluid_vol *= -1.0 * p_fluid_AD * det_F_AD;

        //Get some info from constitutive model of fluid
        const ADNumberType det_Fve_aux = lqph[q_point]->get_converged_det_Fve();
        const double det_Fve_converged = Tensor<0,dim,double>(det_Fve_aux);
                                         //Needs to be double, not AD number
        const Tensor<1,dim,ADNumberType> overall_body_force
                  = lqph[q_point]->get_overall_body_force(F_AD, parameters);

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            const unsigned int i_group = fe.system_to_base_index(i).first.first;

            if (i_group == u_block)
            {
                residual_ad[i] += symm_grad_Nu[i]*(tau_E+tau_fluid_vol)*JxW;
                residual_ad[i] -= Nu[i]*overall_body_force*JxW;
            }
            else if (i_group == p_fluid_block)
            {
                const Tensor<1,dim,ADNumberType> seepage_vel_current
                  = lqph[q_point]->get_seepage_velocity_current(F_AD, grad_p_fluid_AD);
                residual_ad[i] += Np[i]*(det_Fve_AD-det_Fve_converged)*JxW;
                residual_ad[i] -= time.get_delta_t()*grad_Np[i]
                                   *seepage_vel_current*JxW;
            }
            else
              Assert(i_group <= p_fluid_block, ExcInternalError());
          }
      }

      // Assemble the Neumann contribution (external force contribution).
      //Loop over faces in element
      for (unsigned int face =0; face<GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary() == true)
        {
          scratch.fe_face_values_ref.reinit(cell, face);

          for (unsigned int f_q_point=0; f_q_point<n_q_points_f; ++f_q_point)
          {
            const Tensor<1,dim> &N
                      = scratch.fe_face_values_ref.normal_vector(f_q_point);
            const Point<dim>    &pt
                   = scratch.fe_face_values_ref.quadrature_point(f_q_point);
            const Tensor<1,dim> traction
             = get_neumann_traction(cell->face(face)->boundary_id(), pt, N);
            const double flow
            = get_prescribed_fluid_flow(cell->face(face)->boundary_id(),pt);

            if ((traction.norm()<1e-12) && (std::abs(flow)<1e-12)) continue;

            const double JxW_f = scratch.fe_face_values_ref.JxW(f_q_point);

            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const unsigned int i_group = fe.system_to_base_index(i).first.first;

              if ((i_group == u_block) && (traction.norm()>1e-12))
              {
                  const unsigned int component_i =
                                       fe.system_to_component_index(i).first;
                  const double Nu_f =
                        scratch.fe_face_values_ref.shape_value(i, f_q_point);
                  residual_ad[i] -= (Nu_f * traction[component_i]) * JxW_f;
              }
              if ((i_group == p_fluid_block) && (std::abs(flow)>1e-12))
              {
                  const double Nu_p =
                        scratch.fe_face_values_ref.shape_value(i, f_q_point);
                  residual_ad[i] -= (Nu_p * flow) * JxW_f;
              }
            }
          }
        }
    }

    // Linearise the residual
    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
        const ADNumberType &R_i = residual_ad[i];

        data.cell_rhs(i) -= R_i.val();
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          data.cell_matrix(i,j) += R_i.fastAccessDx(j);
    }
}

//Store the converged values of the internal variables
template <int dim>
void Solid<dim>::update_end_timestep()
{
      FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
      cell (IteratorFilters::LocallyOwnedCell(),
            dof_handler_ref.begin_active()),
      endc (IteratorFilters::LocallyOwnedCell(),
            dof_handler_ref.end());
      for (; cell!=endc; ++cell)
      {
        Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
        const std::vector<std::shared_ptr<PointHistory<dim,ADNumberType>>>
            lqph = quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          lqph[q_point]->update_end_timestep();
      }
}


 //Solve the linearized equations
 template <int dim>
 void Solid<dim>::solve_linear_system
    (TrilinosWrappers::MPI::BlockVector &newton_update_OUT)
 {

       timerconsole.enter_subsection("Linear solver");
       timerfile.enter_subsection("Linear solver");
       pcout   << " SLV " << std::flush;
       outfile << " SLV " << std::flush;

       TrilinosWrappers::MPI::Vector newton_update_nb;
       newton_update_nb.reinit(locally_owned_dofs, mpi_communicator);

       SolverControl solver_control(tangent_matrix_nb.m(),
                                    1.0e-6*system_rhs_nb.l2_norm());
       TrilinosWrappers::SolverDirect solver(solver_control);
       solver.solve(tangent_matrix_nb, newton_update_nb, system_rhs_nb);

       // Copy the non-block solution back to block system
       for (unsigned int i=0; i<locally_owned_dofs.n_elements(); ++i)
         {
           const types::global_dof_index idx_i
                                    = locally_owned_dofs.nth_index_in_set(i);
           newton_update_OUT(idx_i) = newton_update_nb(idx_i);
         }
       newton_update_OUT.compress(VectorOperation::insert);

       timerconsole.leave_subsection();
       timerfile.leave_subsection();
 }

//Class to be able to output results correctly when using Paraview
template<int dim, class DH=DoFHandler<dim> >
class FilteredDataOut : public DataOut<dim, DH>
{
    public:
      FilteredDataOut (const unsigned int subdomain_id)
        :
        subdomain_id (subdomain_id)
      {}

      virtual ~FilteredDataOut() {}

      virtual typename DataOut<dim, DH>::cell_iterator
      first_cell()
      {
        typename DataOut<dim, DH>::cell_iterator
        cell = this->dofs->begin_active();
        while ( (cell!=this->dofs->end()) &&
                (cell->subdomain_id()!=subdomain_id) )
          ++cell;
        return cell;
      }

      virtual typename DataOut<dim, DH>::cell_iterator
      next_cell(const typename DataOut<dim, DH>::cell_iterator &old_cell)
      {
        if (old_cell!=this->dofs->end())
        {
            const IteratorFilters::SubdomainEqualTo predicate(subdomain_id);
            return
              ++(FilteredIterator<typename DataOut<dim,DH>::cell_iterator>
                 (predicate,old_cell));
        }
        else
          return old_cell;
      }

    private:
      const unsigned int subdomain_id;
};

//Class to be able to output results correctly when using Paraview
template<int dim, class DH=DoFHandler<dim> >
class FilteredDataOutFaces : public DataOutFaces<dim,DH>
{
    public:
      FilteredDataOutFaces (const unsigned int subdomain_id)
        :
        subdomain_id (subdomain_id)
      {}

      virtual ~FilteredDataOutFaces() {}

      virtual typename DataOutFaces<dim,DH>::cell_iterator
      first_cell ()
      {
        typename DataOutFaces<dim,DH>::cell_iterator
        cell = this->dofs->begin_active();
        while ((cell!=this->dofs->end()) && (cell->subdomain_id()!=subdomain_id))
          ++cell;
        return cell;
      }

      virtual typename DataOutFaces<dim,DH>::cell_iterator
      next_cell (const typename DataOutFaces<dim, DH>::cell_iterator &old_cell)
      {
        if (old_cell!=this->dofs->end())
        {
            const IteratorFilters::SubdomainEqualTo predicate(subdomain_id);
            return
              ++(FilteredIterator<typename DataOutFaces<dim,DH>::cell_iterator>
                 (predicate,old_cell));
        }
        else
          return old_cell;
      }

    private:
      const unsigned int subdomain_id;
};

//Class to compute gradient of the pressure
template <int dim>
class GradientPostprocessor : public DataPostprocessorVector<dim>
{
    public:
      GradientPostprocessor (const unsigned int p_fluid_component)
        :
        DataPostprocessorVector<dim>("grad_p",
                                     update_gradients),
        p_fluid_component (p_fluid_component)
      {}

      virtual ~GradientPostprocessor(){}

      virtual void evaluate_vector_field
          (const DataPostprocessorInputs::Vector<dim> &input_data,
           std::vector<Vector<double>>                &computed_quantities) const
      {
        AssertDimension (input_data.solution_gradients.size(),
                         computed_quantities.size());
        for (unsigned int p=0; p<input_data.solution_gradients.size(); ++p)
          {
            AssertDimension (computed_quantities[p].size(), dim);
            for (unsigned int d=0; d<dim; ++d)
              computed_quantities[p][d]
                = input_data.solution_gradients[p][p_fluid_component][d];
          }
      }

    private:
      const unsigned int  p_fluid_component;
};

//Print results to vtu file
template <int dim> void Solid<dim>::output_results_to_vtu
                      (const unsigned int timestep,
                       const double current_time,
                       TrilinosWrappers::MPI::BlockVector solution_IN) const
{
  TrilinosWrappers::MPI::BlockVector solution_total(locally_owned_partitioning,
                                                    locally_relevant_partitioning,
                                                    mpi_communicator,
                                                    false);
  solution_total = solution_IN;
  Vector<double> material_id;
  material_id.reinit(triangulation.n_active_cells());
  std::vector<types::subdomain_id> partition_int(triangulation.n_active_cells());
  GradientPostprocessor<dim> gradient_postprocessor(p_fluid_component);

   //Declare local variables with number of stress components
   //& assign value according to "dim" value
   unsigned int num_comp_symm_tensor = 6;

  //Declare local vectors to store values
  // OUTPUT AVERAGED ON ELEMENTS -------------------------------------------
  std::vector<Vector<double>>
    cauchy_stresses_total_elements(num_comp_symm_tensor,
                                   Vector<double> (triangulation.n_active_cells()));
  std::vector<Vector<double>>
    cauchy_stresses_E_elements(num_comp_symm_tensor,
                               Vector<double> (triangulation.n_active_cells()));
  std::vector<Vector<double>>
    stretches_elements (dim,
                        Vector<double> (triangulation.n_active_cells()));
  std::vector<Vector<double>>
    seepage_velocity_elements (dim,
                               Vector<double> (triangulation.n_active_cells()));
  Vector<double> porous_dissipation_elements(triangulation.n_active_cells());
  Vector<double> viscous_dissipation_elements(triangulation.n_active_cells());
  Vector<double> solid_vol_fraction_elements(triangulation.n_active_cells());
  Vector<double> growth_stretch_elements(triangulation.n_active_cells());
  Vector<double> div_seepage_velocity_elements(triangulation.n_active_cells());
  Vector<double> norm_seepage_velocity_elements(triangulation.n_active_cells());

  // OUTPUT AVERAGED ON NODES ----------------------------------------------
  // We need to create a new FE space with a single dof per node to avoid
  // duplication of the output on nodes for our problem with dim+1 dofs.
  FE_Q<dim> fe_vertex(1);
  DoFHandler<dim> vertex_handler_ref(triangulation);
  vertex_handler_ref.distribute_dofs(fe_vertex);
  AssertThrow(vertex_handler_ref.n_dofs() == triangulation.n_vertices(),
    ExcDimensionMismatch(vertex_handler_ref.n_dofs(),
                         triangulation.n_vertices()));

  Vector<double> counter_on_vertices_mpi(vertex_handler_ref.n_dofs());
  Vector<double> sum_counter_on_vertices(vertex_handler_ref.n_dofs());

  std::vector<Vector<double>>cauchy_stresses_total_vertex_mpi
                            (num_comp_symm_tensor,
                             Vector<double>(vertex_handler_ref.n_dofs()));
  std::vector<Vector<double>>sum_cauchy_stresses_total_vertex
                            (num_comp_symm_tensor,
                             Vector<double>(vertex_handler_ref.n_dofs()));
  std::vector<Vector<double>>cauchy_stresses_E_vertex_mpi
                            (num_comp_symm_tensor,
                             Vector<double>(vertex_handler_ref.n_dofs()));
  std::vector<Vector<double>>sum_cauchy_stresses_E_vertex
                            (num_comp_symm_tensor,
                             Vector<double>(vertex_handler_ref.n_dofs()));
  std::vector<Vector<double>>stretches_vertex_mpi
                            (dim,
                             Vector<double>(vertex_handler_ref.n_dofs()));
  std::vector<Vector<double>>sum_stretches_vertex
                            (dim,
                             Vector<double>(vertex_handler_ref.n_dofs()));
  Vector<double> porous_dissipation_vertex_mpi(vertex_handler_ref.n_dofs());
  Vector<double> sum_porous_dissipation_vertex(vertex_handler_ref.n_dofs());
  Vector<double> viscous_dissipation_vertex_mpi(vertex_handler_ref.n_dofs());
  Vector<double> sum_viscous_dissipation_vertex(vertex_handler_ref.n_dofs());
  Vector<double> solid_vol_fraction_vertex_mpi(vertex_handler_ref.n_dofs());
  Vector<double> sum_solid_vol_fraction_vertex(vertex_handler_ref.n_dofs());
  Vector<double> growth_stretch_vertex_mpi(vertex_handler_ref.n_dofs());
  Vector<double> sum_growth_stretch_vertex(vertex_handler_ref.n_dofs());
  Vector<double> div_seepage_velocity_vertex_mpi(vertex_handler_ref.n_dofs());
  Vector<double> sum_div_seepage_velocity_vertex(vertex_handler_ref.n_dofs());
  Vector<double> norm_seepage_velocity_vertex_mpi(vertex_handler_ref.n_dofs());
  Vector<double> sum_norm_seepage_velocity_vertex(vertex_handler_ref.n_dofs());

  // We need to create a new FE space with a dim dof per node to
  // be able to ouput data on nodes in vector form
  FESystem<dim> fe_vertex_vec(FE_Q<dim>(1),dim);
  DoFHandler<dim> vertex_vec_handler_ref(triangulation);
  vertex_vec_handler_ref.distribute_dofs(fe_vertex_vec);
  AssertThrow(vertex_vec_handler_ref.n_dofs() == (dim*triangulation.n_vertices()),
    ExcDimensionMismatch(vertex_vec_handler_ref.n_dofs(),
                         (dim*triangulation.n_vertices())));

  Vector<double> seepage_velocity_vertex_vec_mpi(vertex_vec_handler_ref.n_dofs());
  Vector<double> sum_seepage_velocity_vertex_vec(vertex_vec_handler_ref.n_dofs());
  Vector<double> counter_on_vertices_vec_mpi(vertex_vec_handler_ref.n_dofs());
  Vector<double> sum_counter_on_vertices_vec(vertex_vec_handler_ref.n_dofs());
  // -----------------------------------------------------------------------

  //Declare and initialize local unit vectors (to construct tensor basis)
  std::vector<Tensor<1,dim>> basis_vectors (dim, Tensor<1,dim>() );
  for (unsigned int i=0; i<dim; ++i)
      basis_vectors[i][i] = 1;

  //Declare an instance of the material class object
  if (parameters.mat_type == "Neo-Hooke")
      NeoHooke<dim,ADNumberType> material(parameters,time);
  else if (parameters.mat_type == "Ogden")
      Ogden<dim,ADNumberType> material(parameters,time);
  else if (parameters.mat_type == "visco-Ogden")
      visco_Ogden <dim,ADNumberType>material(parameters,time);
  else
      Assert (false, ExcMessage("Material type not implemented"));

  //Define a local instance of FEValues to compute updated values required
  //to calculate stresses
  const UpdateFlags uf_cell(update_values | update_gradients |
                            update_hessians | update_JxW_values);
  FEValues<dim> fe_values_ref (fe, qf_cell, uf_cell);

  //Iterate through elements (cells) and Gauss Points
  FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    cell(IteratorFilters::LocallyOwnedCell(),
         dof_handler_ref.begin_active()),
    endc(IteratorFilters::LocallyOwnedCell(),
         dof_handler_ref.end()),
    cell_v(IteratorFilters::LocallyOwnedCell(),
           vertex_handler_ref.begin_active()),
    cell_v_vec(IteratorFilters::LocallyOwnedCell(),
               vertex_vec_handler_ref.begin_active());
  //start cell loop
  for (; cell!=endc; ++cell, ++cell_v, ++cell_v_vec)
  {
      Assert(cell->is_locally_owned(), ExcInternalError());
      Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

      material_id(cell->active_cell_index())=
         static_cast<int>(cell->material_id());

      fe_values_ref.reinit(cell);

      // Computing solutions, gradients and hessians of unknowns
      std::vector<double> solution_values_p_fluid(n_q_points);
      fe_values_ref[p_fluid_fe].get_function_values(solution_total,
                                                    solution_values_p_fluid);

      std::vector<Tensor<2,dim>> solution_grads_u(n_q_points);
      fe_values_ref[u_fe].get_function_gradients(solution_total,
                                                 solution_grads_u);

      std::vector<Tensor<1,dim>> solution_grads_p_fluid(n_q_points);
      fe_values_ref[p_fluid_fe].get_function_gradients(solution_total,
                                                       solution_grads_p_fluid);

      std::vector<Tensor<2,dim>> solution_hess_p_fluid(n_q_points);
      fe_values_ref[p_fluid_fe].get_function_hessians(solution_total,
                                                      solution_hess_p_fluid);

      std::vector<Tensor<3,dim>> solution_hess_u(n_q_points);
      fe_values_ref[u_fe].get_function_hessians(solution_total,
                                                solution_hess_u);

      // Start gauss point loop
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {
          const Tensor<2,dim,ADNumberType>
            F_AD = Physics::Elasticity::Kinematics::F(solution_grads_u[q_point]);
          ADNumberType det_F_AD = determinant(F_AD);
          const double det_F = Tensor<0,dim,double>(det_F_AD);

          const std::vector<std::shared_ptr<const PointHistory<dim,ADNumberType>>>
              lqph = quadrature_point_history.get_data(cell);
          Assert(lqph.size() == n_q_points, ExcInternalError());

          const double p_fluid = solution_values_p_fluid[q_point];

          // Cauchy stress
          static const SymmetricTensor<2,dim,double>
            I (Physics::Elasticity::StandardTensors<dim>::I);
          SymmetricTensor<2,dim> sigma_E;
          const SymmetricTensor<2,dim,ADNumberType> sigma_E_AD =
            lqph[q_point]->get_Cauchy_E(F_AD);

          for (unsigned int i=0; i<dim; ++i)
              for (unsigned int j=0; j<dim; ++j)
                 sigma_E[i][j] = Tensor<0,dim,double>(sigma_E_AD[i][j]);

          SymmetricTensor<2,dim> sigma_fluid_vol (I);
          sigma_fluid_vol *= -p_fluid;
          const SymmetricTensor<2,dim> sigma = sigma_E + sigma_fluid_vol;

          // Volumes
          const double solid_vol_fraction = (parameters.solid_vol_frac)/det_F;

          // Green-Lagrange strain
          const Tensor<2,dim> E_strain = 0.5*(transpose(F_AD)*F_AD - I);

          // Seepage velocity
          const Tensor<2,dim,ADNumberType> F_inv = invert(F_AD);
          const Tensor<1,dim,ADNumberType> grad_p_fluid_AD =
                                    solution_grads_p_fluid[q_point]*F_inv;
          const Tensor<1,dim,ADNumberType> seepage_vel_AD =
           lqph[q_point]->get_seepage_velocity_current(F_AD, grad_p_fluid_AD);

          // Hessian of pressure in current configuration
          const Tensor<2,dim,ADNumberType> F_inv_T = transpose(F_inv);
          const Tensor<2,dim,ADNumberType> hess_p_fluid_AD =
                                   F_inv_T*solution_hess_p_fluid[q_point]*F_inv;

          // Hessian of displacements in ref configuration
          const Tensor<3,dim,ADNumberType> hess_u_AD = solution_hess_u[q_point];

          // Gradient of determinat of F
          const Tensor<1,dim,ADNumberType> grad_det_F_AD = det_F_AD * double_contract<0,0,1,1>(F_inv_T, hess_u_AD);
          const ADNumberType div_seepage_vel_AD
              =  lqph[q_point]->get_div_seepage_vel(grad_p_fluid_AD,
                                                    hess_p_fluid_AD,
                                                    grad_det_F_AD,
                                                    F_AD);

          const double div_seepage_vel = Tensor<0,dim,double>(div_seepage_vel_AD);

          // Norm of seepage velocity
          const double norm_seepage_vel = Tensor<0,dim,double>(seepage_vel_AD.norm());

          //Dissipations
          const double porous_dissipation =
            lqph[q_point]->get_porous_dissipation(F_AD, grad_p_fluid_AD);
          const double viscous_dissipation =
            lqph[q_point]->get_viscous_dissipation();

          //Growth
          const double growth_stretch =
            lqph[q_point]->get_converged_growth_stretch();

          // OUTPUT AVERAGED ON ELEMENTS -------------------------------------------
          // Both average on elements and on nodes is NOT weighted with the
          // integration point volume, i.e., we assume equal contribution of each
          // integration point to the average. Ideally, it should be weighted,
          // but I haven't invested time in getting it to work properly.
          if (parameters.outtype == "elements")
          {
              for (unsigned int j=0; j<dim; ++j)
              {
                  cauchy_stresses_total_elements[j](cell->active_cell_index())
                    += ((sigma*basis_vectors[j])*basis_vectors[j])/n_q_points;
                  cauchy_stresses_E_elements[j](cell->active_cell_index())
                    += ((sigma_E*basis_vectors[j])*basis_vectors[j])/n_q_points;
                  stretches_elements[j](cell->active_cell_index())
                    += std::sqrt(1.0+2.0*Tensor<0,dim,double>(E_strain[j][j]))
                       /n_q_points;
                  seepage_velocity_elements[j](cell->active_cell_index())
                    +=  Tensor<0,dim,double>(seepage_vel_AD[j])/n_q_points;
              }
              growth_stretch_elements(cell->active_cell_index())
                += growth_stretch/n_q_points;

              div_seepage_velocity_elements(cell->active_cell_index())
                  += div_seepage_vel/n_q_points;
              norm_seepage_velocity_elements(cell->active_cell_index())
                      += norm_seepage_vel/n_q_points;

              porous_dissipation_elements(cell->active_cell_index())
                +=  porous_dissipation/n_q_points;
              viscous_dissipation_elements(cell->active_cell_index())
                +=  viscous_dissipation/n_q_points;
              solid_vol_fraction_elements(cell->active_cell_index())
                +=  solid_vol_fraction/n_q_points;

              cauchy_stresses_total_elements[3](cell->active_cell_index())
                += ((sigma*basis_vectors[0])*basis_vectors[1])/n_q_points; //sig_xy
              cauchy_stresses_total_elements[4](cell->active_cell_index())
                += ((sigma*basis_vectors[0])*basis_vectors[2])/n_q_points;//sig_xz
              cauchy_stresses_total_elements[5](cell->active_cell_index())
                += ((sigma*basis_vectors[1])*basis_vectors[2])/n_q_points;//sig_yz

              cauchy_stresses_E_elements[3](cell->active_cell_index())
                += ((sigma_E*basis_vectors[0])* basis_vectors[1])/n_q_points; //sig_xy
              cauchy_stresses_E_elements[4](cell->active_cell_index())
                += ((sigma_E*basis_vectors[0])* basis_vectors[2])/n_q_points;//sig_xz
              cauchy_stresses_E_elements[5](cell->active_cell_index())
                += ((sigma_E*basis_vectors[1])* basis_vectors[2])/n_q_points;//sig_yz

          }
          // OUTPUT AVERAGED ON NODES -------------------------------------------
          else if (parameters.outtype == "nodes")
          {
            for (unsigned int v=0; v<(GeometryInfo<dim>::vertices_per_cell); ++v)
            {
                types::global_dof_index local_vertex_indices =
                                              cell_v->vertex_dof_index(v, 0);
                counter_on_vertices_mpi(local_vertex_indices) += 1;
                for (unsigned int k=0; k<dim; ++k)
                {
                    cauchy_stresses_total_vertex_mpi[k](local_vertex_indices)
                      += (sigma*basis_vectors[k])*basis_vectors[k];
                    cauchy_stresses_E_vertex_mpi[k](local_vertex_indices)
                      += (sigma_E*basis_vectors[k])*basis_vectors[k];
                    stretches_vertex_mpi[k](local_vertex_indices)
                      += std::sqrt(1.0+2.0*Tensor<0,dim,double>(E_strain[k][k]));

                    types::global_dof_index local_vertex_vec_indices =
                                          cell_v_vec->vertex_dof_index(v, k);
                    counter_on_vertices_vec_mpi(local_vertex_vec_indices) += 1;
                    seepage_velocity_vertex_vec_mpi(local_vertex_vec_indices)
                      += Tensor<0,dim,double>(seepage_vel_AD[k]);
                }
                growth_stretch_vertex_mpi(local_vertex_indices)
                  += growth_stretch;

                div_seepage_velocity_vertex_mpi(local_vertex_indices)
                  += div_seepage_vel;
                norm_seepage_velocity_vertex_mpi(local_vertex_indices)
                  += norm_seepage_vel;

                porous_dissipation_vertex_mpi(local_vertex_indices)
                  += porous_dissipation;
                viscous_dissipation_vertex_mpi(local_vertex_indices)
                  += viscous_dissipation;
                solid_vol_fraction_vertex_mpi(local_vertex_indices)
                  += solid_vol_fraction;

                cauchy_stresses_total_vertex_mpi[3](local_vertex_indices)
                  += (sigma*basis_vectors[0])*basis_vectors[1]; //sig_xy
                cauchy_stresses_total_vertex_mpi[4](local_vertex_indices)
                  += (sigma*basis_vectors[0])*basis_vectors[2];//sig_xz
                cauchy_stresses_total_vertex_mpi[5](local_vertex_indices)
                  += (sigma*basis_vectors[1])*basis_vectors[2]; //sig_yz

                cauchy_stresses_E_vertex_mpi[3](local_vertex_indices)
                  += (sigma_E*basis_vectors[0])*basis_vectors[1]; //sig_xy
                cauchy_stresses_E_vertex_mpi[4](local_vertex_indices)
                  += (sigma_E*basis_vectors[0])*basis_vectors[2];//sig_xz
                cauchy_stresses_E_vertex_mpi[5](local_vertex_indices)
                  += (sigma_E*basis_vectors[1])*basis_vectors[2]; //sig_yz
              }
        }
        //---------------------------------------------------------------
      } //end gauss point loop
  }//end cell loop

  // Different nodes might have different amount of contributions, e.g.,
  // corner nodes have less integration points contributing to the averaged.
  // This is why we need a counter and divide at the end, outside the cell loop.
  if (parameters.outtype == "nodes")
  {
      for (unsigned int d=0; d<(vertex_handler_ref.n_dofs()); ++d)
        {
          sum_counter_on_vertices[d] =
            Utilities::MPI::sum(counter_on_vertices_mpi[d],
                                mpi_communicator);
          sum_porous_dissipation_vertex[d] =
            Utilities::MPI::sum(porous_dissipation_vertex_mpi[d],
                                mpi_communicator);
          sum_viscous_dissipation_vertex[d] =
            Utilities::MPI::sum(viscous_dissipation_vertex_mpi[d],
                                mpi_communicator);
          sum_solid_vol_fraction_vertex[d] =
            Utilities::MPI::sum(solid_vol_fraction_vertex_mpi[d],
                                mpi_communicator);
          sum_growth_stretch_vertex[d] =
            Utilities::MPI::sum(growth_stretch_vertex_mpi[d],
                                mpi_communicator);
          sum_div_seepage_velocity_vertex[d] =
            Utilities::MPI::sum(div_seepage_velocity_vertex_mpi[d],
                                mpi_communicator);
          sum_norm_seepage_velocity_vertex[d] =
             Utilities::MPI::sum(norm_seepage_velocity_vertex_mpi[d],
                                 mpi_communicator);

          for (unsigned int k=0; k<num_comp_symm_tensor; ++k)
          {
            sum_cauchy_stresses_total_vertex[k][d] =
                Utilities::MPI::sum(cauchy_stresses_total_vertex_mpi[k][d],
                                    mpi_communicator);
            sum_cauchy_stresses_E_vertex[k][d] =
                Utilities::MPI::sum(cauchy_stresses_E_vertex_mpi[k][d],
                                    mpi_communicator);
          }
          for (unsigned int k=0; k<dim; ++k)
          {
            sum_stretches_vertex[k][d] =
                Utilities::MPI::sum(stretches_vertex_mpi[k][d],
                                    mpi_communicator);
          }
        }

        for (unsigned int d=0; d<(vertex_vec_handler_ref.n_dofs()); ++d)
        {
            sum_counter_on_vertices_vec[d] =
                Utilities::MPI::sum(counter_on_vertices_vec_mpi[d],
                                    mpi_communicator);
            sum_seepage_velocity_vertex_vec[d] =
                Utilities::MPI::sum(seepage_velocity_vertex_vec_mpi[d],
                                    mpi_communicator);
        }

        for (unsigned int d=0; d<(vertex_handler_ref.n_dofs()); ++d)
        {
          if (sum_counter_on_vertices[d]>0)
          {
            for (unsigned int i=0; i<num_comp_symm_tensor; ++i)
            {
                sum_cauchy_stresses_total_vertex[i][d] /= sum_counter_on_vertices[d];
                sum_cauchy_stresses_E_vertex[i][d] /= sum_counter_on_vertices[d];
            }
            for (unsigned int i=0; i<dim; ++i)
            {
                sum_stretches_vertex[i][d] /= sum_counter_on_vertices[d];
            }
            sum_porous_dissipation_vertex[d] /= sum_counter_on_vertices[d];
            sum_viscous_dissipation_vertex[d] /= sum_counter_on_vertices[d];
            sum_solid_vol_fraction_vertex[d] /= sum_counter_on_vertices[d];
            sum_growth_stretch_vertex[d] /= sum_counter_on_vertices[d];
            sum_div_seepage_velocity_vertex[d] /= sum_counter_on_vertices[d];
            sum_norm_seepage_velocity_vertex[d] /= sum_counter_on_vertices[d];
          }
        }

        for (unsigned int d=0; d<(vertex_vec_handler_ref.n_dofs()); ++d)
        {
          if (sum_counter_on_vertices_vec[d]>0)
          {
            sum_seepage_velocity_vertex_vec[d] /= sum_counter_on_vertices_vec[d];
          }
        }
  }

  // Add the results to the solution to create the output file for Paraview
  FilteredDataOut<dim> data_out(this_mpi_process);
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    comp_type(dim,
              DataComponentInterpretation::component_is_part_of_vector);
  comp_type.push_back(DataComponentInterpretation::component_is_scalar);

  GridTools::get_subdomain_association(triangulation, partition_int);

  std::vector<std::string> solution_name(dim, "displacement");
  solution_name.push_back("pore_pressure");

  data_out.attach_dof_handler(dof_handler_ref);
  data_out.add_data_vector(solution_total,
                           solution_name,
                           DataOut<dim>::type_dof_data,
                           comp_type);

  data_out.add_data_vector(solution_total,
                           gradient_postprocessor);

  const Vector<double> partitioning(partition_int.begin(),
                                    partition_int.end());

  data_out.add_data_vector(partitioning, "partitioning");
  data_out.add_data_vector(material_id, "material_id");

  // Integration point results -----------------------------------------------------------
  if (parameters.outtype == "elements")
  {
    data_out.add_data_vector(cauchy_stresses_total_elements[0], "cauchy_xx");
    data_out.add_data_vector(cauchy_stresses_total_elements[1], "cauchy_yy");
    data_out.add_data_vector(cauchy_stresses_total_elements[2], "cauchy_zz");
    data_out.add_data_vector(cauchy_stresses_total_elements[3], "cauchy_xy");
    data_out.add_data_vector(cauchy_stresses_total_elements[4], "cauchy_xz");
    data_out.add_data_vector(cauchy_stresses_total_elements[5], "cauchy_yz");

    data_out.add_data_vector(cauchy_stresses_E_elements[0], "cauchy_E_xx");
    data_out.add_data_vector(cauchy_stresses_E_elements[1], "cauchy_E_yy");
    data_out.add_data_vector(cauchy_stresses_E_elements[2], "cauchy_E_zz");
    data_out.add_data_vector(cauchy_stresses_E_elements[3], "cauchy_E_xy");
    data_out.add_data_vector(cauchy_stresses_E_elements[4], "cauchy_E_xz");
    data_out.add_data_vector(cauchy_stresses_E_elements[5], "cauchy_E_yz");

    data_out.add_data_vector(stretches_elements[0], "stretch_xx");
    data_out.add_data_vector(stretches_elements[1], "stretch_yy");
    data_out.add_data_vector(stretches_elements[2], "stretch_zz");

    data_out.add_data_vector(seepage_velocity_elements[0], "seepage_velocity_x");
    data_out.add_data_vector(seepage_velocity_elements[1], "seepage_velocity_y");
    data_out.add_data_vector(seepage_velocity_elements[2], "seepage_velocity_z");

    data_out.add_data_vector(div_seepage_velocity_elements, "div_seepage_velocity");
    data_out.add_data_vector(norm_seepage_velocity_elements, "norm_seepage_velocity");
    data_out.add_data_vector(growth_stretch_elements, "growth_stretch");
    data_out.add_data_vector(porous_dissipation_elements, "porous_dissipation");
    data_out.add_data_vector(viscous_dissipation_elements, "viscous_dissipation");
    data_out.add_data_vector(solid_vol_fraction_elements, "solid_vol_fraction");
  }
  else if  (parameters.outtype == "nodes")
  {
      data_out.add_data_vector(vertex_handler_ref,
                               sum_cauchy_stresses_total_vertex[0],
                               "cauchy_xx");
      data_out.add_data_vector(vertex_handler_ref,
                               sum_cauchy_stresses_total_vertex[1],
                               "cauchy_yy");
      data_out.add_data_vector(vertex_handler_ref,
                               sum_cauchy_stresses_total_vertex[2],
                               "cauchy_zz");
      data_out.add_data_vector(vertex_handler_ref,
                               sum_cauchy_stresses_total_vertex[3],
                               "cauchy_xy");
      data_out.add_data_vector(vertex_handler_ref,
                               sum_cauchy_stresses_total_vertex[4],
                               "cauchy_xz");
      data_out.add_data_vector(vertex_handler_ref,
                               sum_cauchy_stresses_total_vertex[5],
                               "cauchy_yz");

      data_out.add_data_vector(vertex_handler_ref,
                               sum_cauchy_stresses_E_vertex[0],
                               "cauchy_E_xx");
      data_out.add_data_vector(vertex_handler_ref,
                               sum_cauchy_stresses_E_vertex[1],
                               "cauchy_E_yy");
      data_out.add_data_vector(vertex_handler_ref,
                               sum_cauchy_stresses_E_vertex[2],
                               "cauchy_E_zz");
      data_out.add_data_vector(vertex_handler_ref,
                               sum_cauchy_stresses_E_vertex[3],
                               "cauchy_E_xy");
      data_out.add_data_vector(vertex_handler_ref,
                               sum_cauchy_stresses_E_vertex[4],
                               "cauchy_E_xz");
      data_out.add_data_vector(vertex_handler_ref,
                               sum_cauchy_stresses_E_vertex[5],
                               "cauchy_E_yz");

      data_out.add_data_vector(vertex_handler_ref,
                               sum_stretches_vertex[0],
                               "stretch_xx");
      data_out.add_data_vector(vertex_handler_ref,
                               sum_stretches_vertex[1],
                               "stretch_yy");
      data_out.add_data_vector(vertex_handler_ref,
                               sum_stretches_vertex[2],
                               "stretch_zz");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
     comp_type_vec(dim,
                   DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> solution_name_vec(dim,"seepage_velocity");

    data_out.add_data_vector(vertex_vec_handler_ref,
                             sum_seepage_velocity_vertex_vec,
                             solution_name_vec,
                             comp_type_vec);

    data_out.add_data_vector(vertex_handler_ref,
                             sum_div_seepage_velocity_vertex,
                             "div_seepage_velocity");
    data_out.add_data_vector(vertex_handler_ref,
                             sum_norm_seepage_velocity_vertex,
                             "norm_seepage_velocity");
    data_out.add_data_vector(vertex_handler_ref,
                             sum_growth_stretch_vertex,
                             "growth_stretch");
    data_out.add_data_vector(vertex_handler_ref,
                             sum_porous_dissipation_vertex,
                             "porous_dissipation");
    data_out.add_data_vector(vertex_handler_ref,
                             sum_viscous_dissipation_vertex,
                             "viscous_dissipation");
    data_out.add_data_vector(vertex_handler_ref,
                             sum_solid_vol_fraction_vertex,
                             "solid_vol_fraction");
  }
//---------------------------------------------------------------------

  if (parameters.poly_degree_output == 0)
      data_out.build_patches(parameters.poly_degree_displ);
  else
      data_out.build_patches(parameters.poly_degree_output);

  struct Filename
  {
    static std::string get_filename_vtu(unsigned int process,
                                        unsigned int timestep,
                                        const unsigned int n_digits = 5)
    {
      std::ostringstream filename_vtu;
      filename_vtu
      << "solution."
      << Utilities::int_to_string(process, n_digits)
      << "."
      << Utilities::int_to_string(timestep, n_digits)
      << ".vtu";
      return filename_vtu.str();
    }

    static std::string get_filename_pvtu(unsigned int timestep,
                                         const unsigned int n_digits = 5)
    {
      std::ostringstream filename_vtu;
      filename_vtu
      << "solution."
      << Utilities::int_to_string(timestep, n_digits)
      << ".pvtu";
      return filename_vtu.str();
    }

    static std::string get_filename_pvd (void)
    {
      std::ostringstream filename_vtu;
      filename_vtu
      << "solution.pvd";
      return filename_vtu.str();
    }
  };

  const std::string filename_vtu = Filename::get_filename_vtu(this_mpi_process,
                                                              timestep);
  std::ofstream output(filename_vtu.c_str());
  data_out.write_vtu(output);

  // We have a collection of files written in parallel
  // This next set of steps should only be performed by master process
  if (this_mpi_process == 0)
  {
    // List of all files written out at this timestep by all processors
    std::vector<std::string> parallel_filenames_vtu;
    for (unsigned int p=0; p<n_mpi_processes; ++p)
    {
      parallel_filenames_vtu.push_back(Filename::get_filename_vtu(p, timestep));
    }

    const std::string filename_pvtu(Filename::get_filename_pvtu(timestep));
    std::ofstream pvtu_master(filename_pvtu.c_str());
    data_out.write_pvtu_record(pvtu_master,
                               parallel_filenames_vtu);

    // Time dependent data master file
    static std::vector<std::pair<double,std::string>> time_and_name_history;
    time_and_name_history.push_back(std::make_pair(current_time,
                                                    filename_pvtu));
    const std::string filename_pvd(Filename::get_filename_pvd());
    std::ofstream pvd_output(filename_pvd.c_str());
    DataOutBase::write_pvd_record(pvd_output, time_and_name_history);
  }
}
//Print boundary conditions to vtu file
//This function os analogous to the output_results_to_vtu function,
// except that we only print to file the surface information.
template <int dim> void Solid<dim>::output_bcs_to_vtu
                      (const unsigned int timestep,
                       const double current_time,
                       TrilinosWrappers::MPI::BlockVector solution_IN) const
{
  TrilinosWrappers::MPI::BlockVector solution_total(locally_owned_partitioning,
                                                    locally_relevant_partitioning,
                                                    mpi_communicator,
                                                    false);
  solution_total = solution_IN;
  Vector<double> material_id;
  material_id.reinit(triangulation.n_active_cells());

  //Declare local vectors to store values
  // OUTPUT AVERAGED ON ELEMENTS -------------------------------------------
  std::vector<Vector<double>>
    loads_faces(dim,
                Vector<double> (triangulation.n_active_cells()));
  // OUTPUT AVERAGED ON NODES ----------------------------------------------
  FESystem<dim> fe_vertex_vec(FE_Q<dim>(1),dim);
  DoFHandler<dim> vertex_vec_handler_ref(triangulation);
  vertex_vec_handler_ref.distribute_dofs(fe_vertex_vec);
  AssertThrow(vertex_vec_handler_ref.n_dofs()==(dim*triangulation.n_vertices()),
    ExcDimensionMismatch(vertex_vec_handler_ref.n_dofs(),
                         (dim*triangulation.n_vertices())));
  Vector<double> loads_vertex_vec_mpi(vertex_vec_handler_ref.n_dofs());
  Vector<double> sum_loads_vertex_vec(vertex_vec_handler_ref.n_dofs());
  Vector<double> counter_on_vertices_vec_mpi(vertex_vec_handler_ref.n_dofs());
  Vector<double> sum_counter_on_vertices_vec(vertex_vec_handler_ref.n_dofs());
  // -----------------------------------------------------------------------

  //Declare an instance of the material class object
 if (parameters.mat_type == "Neo-Hooke")
     NeoHooke<dim, ADNumberType> material(parameters,time);
 else if (parameters.mat_type == "Ogden")
     Ogden<dim, ADNumberType> material(parameters, time);
 else if (parameters.mat_type == "visco-Ogden")
     visco_Ogden <dim, ADNumberType>material(parameters,time);
  else
      Assert (false, ExcMessage("Material type not implemented"));

  //Iterate through elements (cells) and Gauss Points
  FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    cell(IteratorFilters::LocallyOwnedCell(),
         dof_handler_ref.begin_active()),
    endc(IteratorFilters::LocallyOwnedCell(),
         dof_handler_ref.end()),
    cell_v_vec(IteratorFilters::LocallyOwnedCell(),
               vertex_vec_handler_ref.begin_active());
  //start cell loop
  for (; cell!=endc; ++cell, ++cell_v_vec)
  {
      Assert(cell->is_locally_owned(), ExcInternalError());
      Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

      material_id(cell->active_cell_index())=static_cast<int>(cell->material_id());

      const UpdateFlags uf_face(update_quadrature_points | update_normal_vectors |
                                update_values | update_JxW_values );
      FEFaceValues<dim> fe_face_values_ref(fe, qf_face, uf_face);

      //Start loop over faces in element
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary() == true)
        {
            fe_face_values_ref.reinit(cell, face);

            //start gauss point loop
            for (unsigned int f_q_point=0; f_q_point<n_q_points_f; ++f_q_point)
            {
                // Compute load vectors derived from Neumann bcs on surface
                const Tensor<1,dim> &N
                                = fe_face_values_ref.normal_vector(f_q_point);
                const Point<dim> &pt
                             = fe_face_values_ref.quadrature_point(f_q_point);
                const Tensor<1,dim> traction =
                 get_neumann_traction(cell->face(face)->boundary_id(), pt, N);

                if (traction.norm()<1e-12) continue;
                // OUTPUT AVERAGED ON ELEMENTS -------------------------------------------
                if (parameters.outtype == "elements")
                {
                    for (unsigned int j=0; j<dim; ++j)
                    {
                        loads_faces[j](cell->active_cell_index())
                          += traction[j]/n_q_points_f;
                    }
                }
                // OUTPUT AVERAGED ON NODES -------------------------------------------
                else if (parameters.outtype == "nodes")
                {
                  for (unsigned int v=0; v<(GeometryInfo<dim>::vertices_per_face); ++v)
                  {
                      for (unsigned int k=0; k<dim; ++k)
                      {
                        types::global_dof_index local_vertex_vec_indices
                            = cell_v_vec->face(face)->vertex_dof_index(v, k);
                        counter_on_vertices_vec_mpi(local_vertex_vec_indices) += 1;
                        loads_vertex_vec_mpi(local_vertex_vec_indices) += traction[k];
                      }
                   }
                }
                //--------------------------------------------------------------
              }//end gauss point loop
        }//end if face is in boundary
    }//end face loop
  }//end cell loop

  if (parameters.outtype == "nodes")
  {
      for (unsigned int d=0; d<(vertex_vec_handler_ref.n_dofs()); ++d)
      {
          sum_counter_on_vertices_vec[d] =
            Utilities::MPI::sum(counter_on_vertices_vec_mpi[d],
                                mpi_communicator);
          sum_loads_vertex_vec[d] =
            Utilities::MPI::sum(loads_vertex_vec_mpi[d],
                                mpi_communicator);
      }

      for (unsigned int d=0; d<(vertex_vec_handler_ref.n_dofs()); ++d)
      {
        if (sum_counter_on_vertices_vec[d]>0)
        {
          sum_loads_vertex_vec[d] /= sum_counter_on_vertices_vec[d];
        }
      }
  }

  FilteredDataOutFaces<dim> data_out_face(this_mpi_process);

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    face_comp_type(dim,
                   DataComponentInterpretation::component_is_part_of_vector);
  face_comp_type.push_back(DataComponentInterpretation::component_is_scalar);

  std::vector<std::string> ouput_name_face(dim, "displacement");
  ouput_name_face.push_back("pore_pressure");

  data_out_face.attach_dof_handler(dof_handler_ref);
  data_out_face.add_data_vector(solution_total,
                                ouput_name_face,
                                DataOutFaces<dim>::type_dof_data,
                                face_comp_type);

  // Integration point results -----------------------------------------------------------
  if (parameters.outtype == "elements")
  {
    data_out_face.add_data_vector(loads_faces[0], "load_x");
    data_out_face.add_data_vector(loads_faces[1], "load_y");
    data_out_face.add_data_vector(loads_faces[2], "load_z");
  }
  else if  (parameters.outtype == "nodes")
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
     face_comp_type_vec(dim,
                        DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> ouput_name_face_vec(dim, "load");

    data_out_face.add_data_vector(vertex_vec_handler_ref,
                                  sum_loads_vertex_vec,
                                  ouput_name_face_vec,
                                  face_comp_type_vec);
  }
//---------------------------------------------------------------------

  if (parameters.poly_degree_output == 0)
    data_out_face.build_patches (parameters.poly_degree_displ);
  else
    data_out_face.build_patches (parameters.poly_degree_output);

  struct Filename_faces
  {
    static std::string get_filename_face_vtu(unsigned int process,
                                             unsigned int timestep,
                                             const unsigned int n_digits = 5)
    {
      std::ostringstream filename_face_vtu;
      filename_face_vtu
      << "bcs."
      << Utilities::int_to_string(process, n_digits)
      << "."
      << Utilities::int_to_string(timestep, n_digits)
      << ".vtu";
      return filename_face_vtu.str();
    }

    static std::string get_filename_face_pvtu(unsigned int timestep,
                                              const unsigned int n_digits = 5)
    {
      std::ostringstream filename_face_vtu;
      filename_face_vtu
      << "bcs."
      << Utilities::int_to_string(timestep, n_digits)
      << ".pvtu";
      return filename_face_vtu.str();
    }

    static std::string get_filename_face_pvd(void)
    {
      std::ostringstream filename_face_vtu;
      filename_face_vtu
      << "bcs.pvd";
      return filename_face_vtu.str();
    }
  };

  const std::string filename_face_vtu =
      Filename_faces::get_filename_face_vtu(this_mpi_process, timestep);
  std::ofstream output_face(filename_face_vtu.c_str());
  data_out_face.write_vtu(output_face);

  // We have a collection of files written in parallel
  // This next set of steps should only be performed by master process
  if (this_mpi_process == 0)
  {
    // List of all files written out at this timestep by all processors
    std::vector<std::string> parallel_filenames_face_vtu;
    for (unsigned int p=0; p<n_mpi_processes; ++p)
    {
      parallel_filenames_face_vtu.push_back(
                        Filename_faces::get_filename_face_vtu(p, timestep));
    }

    const std::string filename_face_pvtu (
                        Filename_faces::get_filename_face_pvtu(timestep));
    std::ofstream pvtu_master(filename_face_pvtu.c_str());
    data_out_face.write_pvtu_record(pvtu_master,
                                    parallel_filenames_face_vtu);

    // Time dependent data master file
    static std::vector<std::pair<double,std::string>> time_and_name_history_face;
    time_and_name_history_face.push_back (std::make_pair (current_time,
                                                     filename_face_pvtu));
    const std::string filename_face_pvd (Filename_faces::get_filename_face_pvd());
    std::ofstream pvd_output_face(filename_face_pvd.c_str());
    DataOutBase::write_pvd_record(pvd_output_face, time_and_name_history_face);
  }
}

//Print results to plotting file
template <int dim>
void Solid<dim>::output_results_to_plot(
                          const unsigned int timestep,
                          const double current_time,
                          TrilinosWrappers::MPI::BlockVector solution_IN,
                          std::vector<Point<dim> > &tracked_vertices_IN,
                          std::ofstream &plotpointfile) const
{
  TrilinosWrappers::MPI::BlockVector solution_total(locally_owned_partitioning,
                                                    locally_relevant_partitioning,
                                                    mpi_communicator,
                                                    false);

  (void) timestep;
  solution_total = solution_IN;
  Vector<double> material_id;
  material_id.reinit(triangulation.n_active_cells());
  std::vector<types::subdomain_id> partition_int(triangulation.n_active_cells());

  //Variables needed to print the solution file for plotting
  Point<dim> reaction_force;
  Point<dim> reaction_force_pressure;
  Point<dim> reaction_force_extra;
  double total_fluid_flow = 0.0;
  double total_porous_dissipation = 0.0;
  double total_viscous_dissipation = 0.0;
  double total_solid_vol = 0.0;
  double total_vol_current = 0.0;
  double total_vol_reference = 0.0;
  std::vector<Point<dim+1>> solution_vertices(tracked_vertices_IN.size());

  //Auxiliar variables needed for mpi processing
  Tensor<1,dim> sum_reaction_mpi;
  Tensor<1,dim> sum_reaction_pressure_mpi;
  Tensor<1,dim> sum_reaction_extra_mpi;
  sum_reaction_mpi = 0.0;
  sum_reaction_pressure_mpi = 0.0;
  sum_reaction_extra_mpi = 0.0;
  double sum_total_flow_mpi = 0.0;
  double sum_porous_dissipation_mpi = 0.0;
  double sum_viscous_dissipation_mpi = 0.0;
  double sum_solid_vol_mpi = 0.0;
  double sum_vol_current_mpi = 0.0;
  double sum_vol_reference_mpi = 0.0;

  //Declare an instance of the material class object
  if (parameters.mat_type == "Neo-Hooke")
      NeoHooke<dim,ADNumberType> material(parameters,time);
  else if (parameters.mat_type == "Ogden")
      Ogden<dim,ADNumberType> material(parameters, time);
  else if (parameters.mat_type == "visco-Ogden")
      visco_Ogden <dim,ADNumberType>material(parameters,time);
  else
  Assert (false, ExcMessage("Material type not implemented"));

  //Define a local instance of FEValues to compute updated values required
  //to calculate stresses
  const UpdateFlags uf_cell(update_values | update_gradients |
                            update_JxW_values);
  FEValues<dim> fe_values_ref (fe, qf_cell, uf_cell);

  //Iterate through elements (cells) and Gauss Points
  FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    cell(IteratorFilters::LocallyOwnedCell(),
         dof_handler_ref.begin_active()),
    endc(IteratorFilters::LocallyOwnedCell(),
         dof_handler_ref.end());
  //start cell loop
  for (; cell!=endc; ++cell)
  {

      if (cell->subdomain_id() != this_mpi_process) continue;
      material_id(cell->active_cell_index())=static_cast<int>(cell->material_id());

      fe_values_ref.reinit(cell);

      std::vector<Tensor<2,dim>> solution_grads_u(n_q_points);
      fe_values_ref[u_fe].get_function_gradients(solution_total,
                                                 solution_grads_u);

      std::vector<double> solution_values_p_fluid_total(n_q_points);
      fe_values_ref[p_fluid_fe].get_function_values(solution_total,
                                                    solution_values_p_fluid_total);

      std::vector<Tensor<1,dim >> solution_grads_p_fluid_AD(n_q_points);
      fe_values_ref[p_fluid_fe].get_function_gradients(solution_total,
                                                       solution_grads_p_fluid_AD);

      //start gauss point loop
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {
          const Tensor<2,dim,ADNumberType>
            F_AD = Physics::Elasticity::Kinematics::F(solution_grads_u[q_point]);
          ADNumberType det_F_AD = determinant(F_AD);
          const double det_F = Tensor<0,dim,double>(det_F_AD);

          const std::vector<std::shared_ptr<const PointHistory<dim,ADNumberType>>>
              lqph = quadrature_point_history.get_data(cell);
          Assert(lqph.size() == n_q_points, ExcInternalError());

          double JxW = fe_values_ref.JxW(q_point);

          const Tensor<2,dim> Fg = lqph[q_point]->get_non_converged_growth_tensor();
          const double det_Fg = determinant(Fg);

          //Volumes
          sum_vol_current_mpi  += det_F * JxW;
          sum_vol_reference_mpi += JxW;
          sum_solid_vol_mpi += parameters.solid_vol_frac * JxW * det_Fg;

          //Seepage velocity
          const Tensor<2,dim,ADNumberType> F_inv = invert(F_AD);
          const Tensor<1,dim,ADNumberType>
            grad_p_fluid_AD =  solution_grads_p_fluid_AD[q_point]*F_inv;
          const Tensor<1,dim,ADNumberType> seepage_vel_AD
          = lqph[q_point]->get_seepage_velocity_current(F_AD, grad_p_fluid_AD);

          //Dissipations
          const double porous_dissipation =
            lqph[q_point]->get_porous_dissipation(F_AD, grad_p_fluid_AD);
          sum_porous_dissipation_mpi += porous_dissipation * det_F * JxW;

          const double viscous_dissipation = lqph[q_point]->get_viscous_dissipation();
          sum_viscous_dissipation_mpi += viscous_dissipation * det_F * JxW;

        //---------------------------------------------------------------
      } //end gauss point loop

      // Compute reaction force on load boundary & total fluid flow across
      // drained boundary.
      // Define a local instance of FEFaceValues to compute values required
      // to calculate reaction force
      const UpdateFlags uf_face( update_values | update_gradients |
                                 update_normal_vectors | update_JxW_values );
      FEFaceValues<dim> fe_face_values_ref(fe, qf_face, uf_face);

      //start face loop
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
      {
          //Reaction force
          if (cell->face(face)->at_boundary() == true &&
              (cell->face(face)->boundary_id() ==
                 get_reaction_boundary_id_for_output().first ||
               cell->face(face)->boundary_id() ==
                 get_reaction_boundary_id_for_output().second  ) )
          {
              fe_face_values_ref.reinit(cell, face);

              //Get displacement gradients for current face
              std::vector<Tensor<2,dim> > solution_grads_u_f(n_q_points_f);
              fe_face_values_ref[u_fe].get_function_gradients
                                                   (solution_total,
                                                    solution_grads_u_f);

              //Get pressure for current element
              std::vector< double > solution_values_p_fluid_total_f(n_q_points_f);
              fe_face_values_ref[p_fluid_fe].get_function_values
                                         (solution_total,
                                          solution_values_p_fluid_total_f);

              //start gauss points on faces loop
              for (unsigned int f_q_point=0; f_q_point<n_q_points_f; ++f_q_point)
              {
                  const Tensor<1,dim> &N = fe_face_values_ref.normal_vector(f_q_point);
                  const double JxW_f = fe_face_values_ref.JxW(f_q_point);

                  //Compute deformation gradient from displacements gradient
                  //(present configuration)
                  const Tensor<2,dim,ADNumberType> F_AD =
                    Physics::Elasticity::Kinematics::F(solution_grads_u_f[f_q_point]);

                  const std::vector<std::shared_ptr<const PointHistory<dim,ADNumberType>>>
                      lqph = quadrature_point_history.get_data(cell);
                  Assert(lqph.size() == n_q_points, ExcInternalError());

                  const double p_fluid = solution_values_p_fluid_total[f_q_point];

                  //Cauchy stress
                  static const SymmetricTensor<2,dim,double>
                    I (Physics::Elasticity::StandardTensors<dim>::I);
                  SymmetricTensor<2,dim> sigma_E;
                  const SymmetricTensor<2,dim,ADNumberType> sigma_E_AD =
                    lqph[f_q_point]->get_Cauchy_E(F_AD);

                  for (unsigned int i=0; i<dim; ++i)
                      for (unsigned int j=0; j<dim; ++j)
                         sigma_E[i][j] = Tensor<0,dim,double>(sigma_E_AD[i][j]);

                  SymmetricTensor<2,dim> sigma_fluid_vol(I);
                  sigma_fluid_vol *= -1.0*p_fluid;
                  const SymmetricTensor<2,dim> sigma = sigma_E+sigma_fluid_vol;
                  sum_reaction_mpi += sigma * N * JxW_f;
                  sum_reaction_pressure_mpi += sigma_fluid_vol * N * JxW_f;
                  sum_reaction_extra_mpi += sigma_E * N * JxW_f;
              }//end gauss points on faces loop
          }

          //Fluid flow
          if (cell->face(face)->at_boundary() == true &&
             (cell->face(face)->boundary_id() ==
                get_drained_boundary_id_for_output().first ||
              cell->face(face)->boundary_id() ==
                get_drained_boundary_id_for_output().second ) )
          {
              fe_face_values_ref.reinit(cell, face);

              //Get displacement gradients for current face
              std::vector<Tensor<2,dim>> solution_grads_u_f(n_q_points_f);
              fe_face_values_ref[u_fe].get_function_gradients
                                                      (solution_total,
                                                       solution_grads_u_f);

              //Get pressure gradients for current face
              std::vector<Tensor<1,dim>> solution_grads_p_f(n_q_points_f);
              fe_face_values_ref[p_fluid_fe].get_function_gradients
                                                       (solution_total,
                                                        solution_grads_p_f);

              //start gauss points on faces loop
              for (unsigned int f_q_point=0; f_q_point<n_q_points_f; ++f_q_point)
              {
                  const Tensor<1,dim> &N =
                            fe_face_values_ref.normal_vector(f_q_point);
                  const double JxW_f = fe_face_values_ref.JxW(f_q_point);

                  //Deformation gradient and inverse from displacements gradient
                  //(present configuration)
                  const Tensor<2,dim,ADNumberType> F_AD
                      = Physics::Elasticity::Kinematics::F(solution_grads_u_f[f_q_point]);

                  const Tensor<2,dim,ADNumberType> F_inv_AD = invert(F_AD);
                  ADNumberType det_F_AD = determinant(F_AD);

                  const std::vector<std::shared_ptr<const PointHistory<dim,ADNumberType>>>
                      lqph = quadrature_point_history.get_data(cell);
                  Assert(lqph.size() == n_q_points, ExcInternalError());

                  //Seepage velocity
                  Tensor<1,dim> seepage;
                  double det_F = Tensor<0,dim,double>(det_F_AD);
                  const Tensor<1,dim,ADNumberType> grad_p
                                    = solution_grads_p_f[f_q_point]*F_inv_AD;
                  const Tensor<1,dim,ADNumberType> seepage_AD
                    = lqph[f_q_point]->get_seepage_velocity_current(F_AD, grad_p);

                  for (unsigned int i=0; i<dim; ++i)
                      seepage[i] = Tensor<0,dim,double>(seepage_AD[i]);

                  sum_total_flow_mpi += (seepage/det_F) * N * JxW_f;
              }//end gauss points on faces loop
          }
      }//end face loop
  }//end cell loop

  //Sum the results from different MPI process and then add to the reaction_force vector
  //In theory, the solution on each surface (each cell) only exists in one MPI process
  //so, we add all MPI process, one will have the solution and the others will be zero
  for (unsigned int d=0; d<dim; ++d)
  {
      reaction_force[d] = Utilities::MPI::sum(sum_reaction_mpi[d],
                                              mpi_communicator);
      reaction_force_pressure[d] = Utilities::MPI::sum(sum_reaction_pressure_mpi[d],
                                                       mpi_communicator);
      reaction_force_extra[d] = Utilities::MPI::sum(sum_reaction_extra_mpi[d],
                                                    mpi_communicator);
  }

  //Same for total fluid flow, and for porous and viscous dissipations
  total_fluid_flow = Utilities::MPI::sum(sum_total_flow_mpi,
                                         mpi_communicator);
  total_porous_dissipation = Utilities::MPI::sum(sum_porous_dissipation_mpi,
                                                 mpi_communicator);
  total_viscous_dissipation = Utilities::MPI::sum(sum_viscous_dissipation_mpi,
                                                  mpi_communicator);
  total_solid_vol = Utilities::MPI::sum(sum_solid_vol_mpi,
                                        mpi_communicator);
  total_vol_current = Utilities::MPI::sum(sum_vol_current_mpi,
                                          mpi_communicator);
  total_vol_reference = Utilities::MPI::sum(sum_vol_reference_mpi,
                                            mpi_communicator);

//  Extract solution for tracked vectors
// Copying an MPI::BlockVector into MPI::Vector is not possible,
// so we copy each block of MPI::BlockVector into an MPI::Vector
// And then we copy the MPI::Vector into "normal" Vectors
  TrilinosWrappers::MPI::Vector solution_vector_u_MPI(solution_total.block(u_block));
  TrilinosWrappers::MPI::Vector solution_vector_p_MPI(solution_total.block(p_fluid_block));
  Vector<double> solution_u_vector(solution_vector_u_MPI);
  Vector<double> solution_p_vector(solution_vector_p_MPI);

  if (this_mpi_process == 0)
  {
      //Append the pressure solution vector to the displacement solution vector,
      //creating a single solution vector equivalent to the original BlockVector
      //so FEFieldFunction will work with the dof_handler_ref.
      Vector<double> solution_vector(solution_p_vector.size()
                                     +solution_u_vector.size());

      for (unsigned int d=0; d<(solution_u_vector.size()); ++d)
          solution_vector[d] = solution_u_vector[d];

      for (unsigned int d=0; d<(solution_p_vector.size()); ++d)
          solution_vector[solution_u_vector.size()+d] = solution_p_vector[d];

      Functions::FEFieldFunction<dim,DoFHandler<dim>,Vector<double>>
      find_solution(dof_handler_ref, solution_vector);

      for (unsigned int p=0; p<tracked_vertices_IN.size(); ++p)
      {
          Vector<double> update(dim+1);
          Point<dim> pt_ref;

          pt_ref[0]= tracked_vertices_IN[p][0];
          pt_ref[1]= tracked_vertices_IN[p][1];
          pt_ref[2]= tracked_vertices_IN[p][2];

         find_solution.vector_value(pt_ref, update);

         for (unsigned int d=0; d<(dim+1); ++d)
         {
             //For values close to zero, set to 0.0
             if (abs(update[d])<1.5*parameters.tol_u)
                 update[d] = 0.0;
             solution_vertices[p][d] = update[d];
         }
      }
// Write the results to the plotting file.
      plotpointfile <<  std::setprecision(6) << std::scientific;
      plotpointfile << std::setw(16) << current_time        << ","
                    << std::setw(15) << total_vol_reference << ","
                    << std::setw(15) << total_vol_current   << ","
                    << std::setw(15) << total_solid_vol     << ",";

      if (current_time == 0.0)
      {
          for (unsigned int p=0; p<tracked_vertices_IN.size(); ++p)
          {
              for (unsigned int d=0; d<dim; ++d)
                  plotpointfile << std::setw(15) << 0.0 << ",";

              plotpointfile << std::setw(15) << parameters.drained_pressure << ",";
          }
          for (unsigned int d=0; d<(3*dim+2); ++d)
              plotpointfile << std::setw(15) << 0.0 << ",";

          plotpointfile << std::setw(15) << 0.0;
      }
      else
      {
          for (unsigned int p=0; p<tracked_vertices_IN.size(); ++p)
              for (unsigned int d=0; d<(dim+1); ++d)
                  plotpointfile << std::setw(15) << solution_vertices[p][d]<< ",";

          for (unsigned int d=0; d<dim; ++d)
              plotpointfile << std::setw(15) << reaction_force[d] << ",";

          for (unsigned int d=0; d<dim; ++d)
              plotpointfile << std::setw(15) << reaction_force_pressure[d] << ",";

          for (unsigned int d=0; d<dim; ++d)
              plotpointfile << std::setw(15) << reaction_force_extra[d] << ",";

          plotpointfile << std::setw(15) << total_fluid_flow << ","
                        << std::setw(15) << total_porous_dissipation<< ","
                        << std::setw(15) << total_viscous_dissipation;
      }
      plotpointfile << std::endl;
  }
}

//Header for console output file
template <int dim>
void Solid<dim>::print_console_file_header(std::ofstream &outputfile) const
{
  outputfile << "/*-----------------------------------------------------------------------------------------";
  outputfile << "\n\n  CompLimb-biomech formulation to model the influence of local mechanical stimuli";
  outputfile << "\n  on joint shape using deal.ii";
  outputfile << "\n\n  Problem setup by E Comellas, ";
  outputfile << "\n  Northeastern University and Universitat Politècnica de Catalunya, 2020";
  outputfile << "\n\n/*-----------------------------------------------------------------------------------------";
  outputfile << "\n\nCONSOLE OUTPUT: \n\n";
}

//Header for plotting output file
template <int dim>
void Solid<dim>::print_plot_file_header(std::vector<Point<dim> > &tracked_vertices,
                                        std::ofstream &plotpointfile) const
{
        plotpointfile << "#\n# *** Solution history for tracked vertices -- DOF: 0 = Ux,  1 = Uy,  2 = Uz,  3 = P ***"
                      << std::endl;

        for  (unsigned int p=0; p<tracked_vertices.size(); ++p)
        {
            plotpointfile << "#        Point " << p << " coordinates:  ";
            for (unsigned int d=0; d<dim; ++d)
              {
                plotpointfile << tracked_vertices[p][d];
                if (!( (p == tracked_vertices.size()-1) && (d == dim-1) ))
                    plotpointfile << ",        ";
              }
            plotpointfile << std::endl;
        }
        plotpointfile << "#    The reaction force is the integral over the loaded surfaces in the "
                      << "undeformed configuration of the Cauchy stress times the normal surface unit vector.\n"
                      << "#    reac(p) corresponds to the volumetric part of the Cauchy stress due to the pore fluid pressure"
                      << " and reac(E) corresponds to the extra part of the Cauchy stress due to the solid contribution."
                      << std::endl
                      << "#    The fluid flow is the integral over the drained surfaces in the "
                      << "undeformed configuration of the seepage velocity times the normal surface unit vector."
                      << std::endl
                      << "# Column number:"
                      << std::endl
                      << "#";

      unsigned int columns = 24;
      for (unsigned int d=1; d<columns; ++d)
          plotpointfile << std::setw(15)<< d <<",";

        plotpointfile << std::setw(15)<< columns
                      << std::endl
                      << "#"
                      << std::right << std::setw(16) << "Time,"
                      << std::right << std::setw(16) << "ref vol,"
                      << std::right << std::setw(16) << "def vol,"
                      << std::right << std::setw(16) << "solid vol,";
        for (unsigned int p=0; p<tracked_vertices.size(); ++p)
            for (unsigned int d=0; d<(dim+1); ++d)
                plotpointfile << std::right << std::setw(11)
                              <<"P" << p << "[" << d << "],";

        for (unsigned int d=0; d<dim; ++d)
            plotpointfile << std::right << std::setw(13)
                          << "reaction [" << d << "],";

        for (unsigned int d=0; d<dim; ++d)
            plotpointfile << std::right << std::setw(13)
                          << "reac(p) [" << d << "],";

        for (unsigned int d=0; d<dim; ++d)
            plotpointfile << std::right << std::setw(13)
                          << "reac(E) [" << d << "],";

        plotpointfile << std::right << std::setw(16) << "fluid flow,"
                      << std::right << std::setw(16) << "porous dissip,"
                      << std::right << std::setw(15) << "viscous dissip"
                      << std::endl;
}

//Footer for console output file
template <int dim>
void Solid<dim>::print_console_file_footer(std::ofstream &outputfile) const
{
       //Copy "parameters" file at end of output file.
       std::ifstream infile("parameters.prm");
       std::string content = "";
       int i;

       for(i=0 ; infile.eof()!=true ; i++)
       {
           char aux = infile.get();
           content += aux;
           if(aux=='\n') content += '#';
       }

       i--;
       content.erase(content.end()-1);
       infile.close();

       outputfile << "\n\n\n\n PARAMETERS FILE USED IN THIS COMPUTATION: \n#"
                  << std::endl
                  << content;
}

//Footer for plotting output file
template <int dim>
void Solid<dim>::print_plot_file_footer(std::ofstream &plotpointfile) const
{
       //Copy "parameters" file at end of output file.
       std::ifstream infile("parameters.prm");
       std::string content = "";
       int i;

       for(i=0 ; infile.eof()!=true ; i++)
       {
           char aux = infile.get();
           content += aux;
           if(aux=='\n') content += '#';
       }

       i--;
       content.erase(content.end()-1);
       infile.close();

       plotpointfile << "#"<< std::endl
                     << "#"<< std::endl
                     << "# PARAMETERS FILE USED IN THIS COMPUTATION:" << std::endl
                     << "#"<< std::endl
                     << content;
}

// @sect3{Continuum growth examples}
// We group the definition of the geometry, boundary and loading conditions
// specific to the examples related to continuum growth into specific classes.

//@sect4{Muffin}
struct TransfMuffin
{
  Point<3> operator()(const Point<3> &in) const
  {
    double radius_incr=1.0;
    double half_height = 0.2;

    Point<3> out;
    out[2] = in[2]+half_height;
    out[0] = in[0]*(1+radius_incr*out[2]/(2*half_height));
    out[1] = in[1]*(1+radius_incr*out[2]/(2*half_height));

    return out;
  }
};

template <int dim>
  class GrowingMuffin
      : public Solid<dim>
{
public:
    GrowingMuffin (const Parameters::AllParameters &parameters)
    : Solid<dim> (parameters)
  {}

  virtual ~GrowingMuffin () {}

private:
  virtual void
  make_grid()
  {

      GridGenerator::cylinder(this->triangulation,
                              0.2,   //radius
                              0.2);  //half-length
      //Create a cylinder around the x-axis. The cylinder extends from
      //x=-"half_length" to x=+"half_length" and its projection into the
      //yz-plane is a circle of radius "radius".
      //The boundaries are colored according to the following scheme:
      //0 for the hull of the cylinder,
      //1 for the left hand face and
      //2 for the right hand face.

       /*
     GridGenerator::truncated_cone(this->triangulation,
                                   0.3,   //radius_0
                                   0.5,   //radius_1
                                   0.25);  //half-length
       */
      //Create a cylinder around the x-axis. The cylinder extends from x=-"half_length"
      //to x=+"half_length" and its projection into the yz-plane is a circle of radius "radius_0"
      //at x=-"half_length" and a circle of radius "radius_1" at x=+"half_length".
      //In between the radius is linearly decreasing.
      //The boundaries are colored according to the following scheme:
      //0 for the hull of the cylinder, 1 for the left hand face and 2 for the right hand face.

     //Rotate cylinder so that it is aligned with the z-axis
     const double rot_angle = 3.0*numbers::PI/2.0;
     GridTools::rotate( rot_angle, 1, this->triangulation);
     // Hull of cylinder = drained boundary        --> 0
     // Left hand face is now bottom face = fixed  --> 1
     // Right hand face is now top face = load     --> 2

     //Shift cylinder upwards in z direction and transform into cone.
     GridTools::transform(TransfMuffin(), this->triangulation);

     this->triangulation.reset_manifold(0);
     const CylindricalManifold<dim> manifold_description_3d(2);
     this->triangulation.set_manifold (0, manifold_description_3d);
     GridTools::scale(this->parameters.scale, this->triangulation);
     this->triangulation.refine_global(std::max(1U,
                                                this->parameters.global_refinement));
     this->triangulation.reset_manifold(0);
  }

  virtual void
  define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices)
  {
    tracked_vertices[0][0] = 0.0*this->parameters.scale;
    tracked_vertices[0][1] = 0.0*this->parameters.scale;
    tracked_vertices[0][2] = 0.25*this->parameters.scale;

    tracked_vertices[1][0] = 0.0*this->parameters.scale;
    tracked_vertices[1][1] = 0.0*this->parameters.scale;
    tracked_vertices[1][2] = 0.0*this->parameters.scale;
  }

  virtual void
  make_dirichlet_constraints(AffineConstraints<double> &constraints)
  {
      if (this->time.get_timestep() < 2)
      {
        VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     2,
                     ConstantFunction<dim>(this->parameters.drained_pressure,
                                           this->n_components),
                     constraints,
                     (this->fe.component_mask(this->pressure)));
      }
      else
      {
        VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     2,
                     ZeroFunction<dim>(this->n_components),
                     constraints,
                     (this->fe.component_mask(this->pressure)));
      }

      VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     1,
                     ZeroFunction<dim>(this->n_components),
                     constraints,
                     (this->fe.component_mask(this->x_displacement) |
                      this->fe.component_mask(this->y_displacement) |
                      this->fe.component_mask(this->z_displacement) ) );

      VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     0,
                     ZeroFunction<dim>(this->n_components),
                     constraints,
                     (this->fe.component_mask(this->x_displacement) |
                      this->fe.component_mask(this->y_displacement) |
                      this->fe.component_mask(this->z_displacement) ) );

   if (this->parameters.load_type == "displacement")
     AssertThrow(false,
        ExcMessage("Displacement loading not defined for the current problem: "
                     + this->parameters.geom_type));
  }

  virtual Tensor<1,dim>
  get_neumann_traction (const types::boundary_id &boundary_id,
                        const Point<dim>         &pt,
                        const Tensor<1,dim>      &N) const
  {

    if (this->parameters.load_type == "pressure")
      AssertThrow(false,
         ExcMessage("Pressure loading not defined for the current problem: "
                     + this->parameters.geom_type));
    //To get rid of warning message
    (void)boundary_id;
    (void)pt;
    (void)N;
    return Tensor<1,dim>();
  }

  virtual double
  get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                             const Point<dim>         &pt) const
  {
      //Silence compiler warnings
      (void)pt;
      (void)boundary_id;
      return 0.0;
  }

  virtual std::pair<types::boundary_id,types::boundary_id>
  get_reaction_boundary_id_for_output() const
  {
      return std::make_pair(0,0);
  }

  virtual std::pair<types::boundary_id,types::boundary_id>
  get_drained_boundary_id_for_output() const
  {
      return std::make_pair(2,2);
  }

  virtual std::pair<double, FEValuesExtractors::Scalar>
  get_dirichlet_load(const types::boundary_id &boundary_id) const
  {
      double displ_incr = 0;
      FEValuesExtractors::Scalar direction;
      (void)boundary_id;
      return std::make_pair(displ_incr,direction);
  }
};

//@sect4{Trapped turtle}
struct TransfTurtle
{
  Point<3> operator()(const Point<3> &in) const
  {
    double x_incr=2.0;

    Point<3> out;
    out[0] = in[0]*(1+x_incr);
    out[1] = in[1];
    out[2] = in[2];

    return out;
  }
};

template <int dim>
  class TrappedTurtle
      : public Solid<dim>
{
public:
    TrappedTurtle (const Parameters::AllParameters &parameters)
    : Solid<dim> (parameters)
  {}

  virtual ~TrappedTurtle () {}

private:
  virtual void
  make_grid()
  {
      const Point< dim > center(0,0,0);
      double radius = 1.0;
      GridGenerator::half_hyper_ball(this->triangulation,
                                     center,
                                     radius );
      //A half hyper-ball around center, which contains 6 in 3d.
      //The cut plane is perpendicular to the x-axis.
      //The boundary indicators are 0 for the curved boundary and 1 for the cut plane.
      //The manifold id for the curved boundary is set to zero, and a SphericalManifold is attached to it.

      //Rotate half-sphere so that it is perpendicular to the z-axis
      const double rot_angle = 3.0*(numbers::PI)/2.0;
      GridTools::rotate(rot_angle, 1, this->triangulation);

      //Elongate half-sphere in the x direction.
      //GridTools::transform(TransfTurtle(), this->triangulation);

     GridTools::scale(this->parameters.scale, this->triangulation);
     this->triangulation.refine_global(std::max(1U,
                                                this->parameters.global_refinement));


     //Set area for constraint
     double cirumf = (numbers::PI)*radius;
     double x_plane_fix = 0.1*this->parameters.scale;
     double margin = (cirumf*this->parameters.scale)
                      /(4*this->parameters.global_refinement);

     typename Triangulation<dim>::active_cell_iterator
     cell = this->triangulation.begin_active(),
     endc = this->triangulation.end();
     for (; cell != endc; ++cell)
     {
       for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
         if ( cell->face(face)->at_boundary() == true  &&
              cell->face(face)->center()[0] > 0        &&
              ( ((abs(cell->face(face)->center()[0]-x_plane_fix)<0.5*margin) &&
                 (abs(cell->face(face)->center()[2])
                    <0.5*radius*this->parameters.scale)) ||
                ((abs(cell->face(face)->center()[0]-x_plane_fix)<0.4*margin) &&
                 (abs(cell->face(face)->center()[2])
                    >0.5*radius*this->parameters.scale))  ) )
               cell->face(face)->set_boundary_id(2);
     }
  }

  virtual void
  define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices)
  {
    tracked_vertices[0][0] = 0.0*this->parameters.scale;
    tracked_vertices[0][1] = 0.0*this->parameters.scale;
    tracked_vertices[0][2] = 1.0*this->parameters.scale;

    tracked_vertices[1][0] = 0.0*this->parameters.scale;
    tracked_vertices[1][1] = 0.0*this->parameters.scale;
    tracked_vertices[1][2] = 0.0*this->parameters.scale;
  }

  virtual void
  make_dirichlet_constraints(AffineConstraints<double> &constraints)
  {
      if (this->time.get_timestep() < 2)
      {
        VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     0,
                     ConstantFunction<dim>(this->parameters.drained_pressure,
                                           this->n_components),
                     constraints,
                     (this->fe.component_mask(this->pressure)));

        VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     1,
                     ConstantFunction<dim>(this->parameters.drained_pressure,
                                           this->n_components),
                     constraints,
                     (this->fe.component_mask(this->pressure)));
      }
      else
      {
        VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     0,
                     ZeroFunction<dim>(this->n_components),
                     constraints,
                     (this->fe.component_mask(this->pressure)));

        VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     1,
                     ZeroFunction<dim>(this->n_components),
                     constraints,
                     (this->fe.component_mask(this->pressure)));
      }
      VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     1,
                     ZeroFunction<dim>(this->n_components),
                     constraints,
                     (this->fe.component_mask(this->z_displacement) ) );

      VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     2,
                     ZeroFunction<dim>(this->n_components),
                     constraints,
                     (this->fe.component_mask(this->x_displacement) |
                      this->fe.component_mask(this->y_displacement) |
                      this->fe.component_mask(this->z_displacement)) );

     if (this->parameters.load_type == "displacement")
       AssertThrow(false,
         ExcMessage("Displacement loading not defined for the current problem: "
                    + this->parameters.geom_type));
  }

  virtual Tensor<1,dim>
  get_neumann_traction (const types::boundary_id &boundary_id,
                        const Point<dim>         &pt,
                        const Tensor<1,dim>      &N) const
  {


   if (this->parameters.load_type == "pressure")
     AssertThrow(false,
          ExcMessage("Pressure loading not defined for the current problem: "
                       + this->parameters.geom_type));

    //To get rid of warning message
    (void)boundary_id;
    (void)pt;
    (void)N;
    return Tensor<1,dim>();
  }

  virtual double
  get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                             const Point<dim>         &pt) const
  {
      //Silence compiler warnings
      (void)pt;
      (void)boundary_id;
      return 0.0;
  }

  virtual std::pair<types::boundary_id,types::boundary_id>
  get_reaction_boundary_id_for_output() const
  {
      return std::make_pair(0,0);
  }

  virtual std::pair<types::boundary_id,types::boundary_id>
  get_drained_boundary_id_for_output() const
  {
      return std::make_pair(2,2);
  }

  virtual std::pair<double, FEValuesExtractors::Scalar>
  get_dirichlet_load(const types::boundary_id &boundary_id) const
  {
      double displ_incr = 0;
      FEValuesExtractors::Scalar direction;
      (void)boundary_id;
      return std::make_pair(displ_incr,direction);
  }
};


//@sect4{Brain cube}
template <int dim>
  class GrowthBaseCube
      : public Solid<dim>
{
public:
    GrowthBaseCube (const Parameters::AllParameters &parameters)
    : Solid<dim> (parameters)
  {}

  virtual ~GrowthBaseCube () {}

private:
  virtual void
  make_grid()
  {
    GridGenerator::hyper_cube(this->triangulation,
                              0.0,
                              1.0,
                              true);
    // Cube 1 x 1 x 1
    // If the colorize flag is true, the boundary_ids of the boundary faces
    // are assigned, such that the lower one in x-direction is 0,
    // the upper one is 1. The indicators for the surfaces in y-direction
    // are 2 and 3, the ones for z are 4 and 5.

    // Assign all faces same boundary id = 0
    typename Triangulation<dim>::active_cell_iterator
    cell = this->triangulation.begin_active(),
    endc = this->triangulation.end();
    for (; cell != endc; ++cell)
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            if (cell->face(face)->at_boundary() == true)
                cell->face(face)->set_boundary_id(0); //All surfaces have boundary id 0

    GridTools::scale(this->parameters.scale, this->triangulation);
    //this->triangulation.refine_global(std::max (1U, this->parameters.global_refinement));
    this->triangulation.refine_global(this->parameters.global_refinement);
  }

  virtual void
  define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices)
  {
    tracked_vertices[0][0] = 0.5*this->parameters.scale;
    tracked_vertices[0][1] = 0.5*this->parameters.scale;
    tracked_vertices[0][2] = 1.0*this->parameters.scale;

    tracked_vertices[1][0] = 0.5*this->parameters.scale;
    tracked_vertices[1][1] = 0.5*this->parameters.scale;
    tracked_vertices[1][2] = 0.5*this->parameters.scale;
  }

  virtual double
  get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                             const Point<dim>         &pt) const
  {
      //Silence compiler warnings
      (void)pt;
      (void)boundary_id;
      return 0.0;
  }

  virtual std::pair<types::boundary_id,types::boundary_id>
  get_reaction_boundary_id_for_output() const
  {
      return std::make_pair(0,0);
  }

  virtual std::pair<types::boundary_id,types::boundary_id>
  get_drained_boundary_id_for_output() const
  {
      return std::make_pair(0,0);
  }

  virtual Tensor<1,dim>
  get_neumann_traction (const types::boundary_id &boundary_id,
                        const Point<dim>         &pt,
                        const Tensor<1,dim>      &N            ) const
  {
    if (this->parameters.load_type == "pressure")
      AssertThrow(false,
          ExcMessage("Pressure loading not defined for the current problem: "
                      + this->parameters.geom_type));

      //To get rid of warning message
      (void)boundary_id;
      (void)pt;
      (void)N;
      return Tensor<1,dim>();
    }

  virtual  std::pair<double, FEValuesExtractors::Scalar>
  get_dirichlet_load(const types::boundary_id &boundary_id) const
  {
      double displ_incr = 0;
      FEValuesExtractors::Scalar direction;
      (void)boundary_id;
      return std::make_pair(displ_incr,direction);
  }
};


template <int dim>
  class GrowthCubeConfinedDrained
      : public GrowthBaseCube<dim>
{
public:
    GrowthCubeConfinedDrained (const Parameters::AllParameters &parameters)
    : GrowthBaseCube<dim> (parameters)
  {}

  virtual ~GrowthCubeConfinedDrained () {}

private:
  virtual void
  make_dirichlet_constraints(AffineConstraints<double> &constraints)
  {
      if (this->time.get_timestep()<2) //Dirichlet BC on pressure nodes
      {
          VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     0,
                     ConstantFunction<dim>(this->parameters.drained_pressure,
                                           this->n_components),
                     constraints,
                     (this->fe.component_mask(this->pressure)));
      }
      else
      {
          VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     0,
                     ZeroFunction<dim>(this->n_components),
                     constraints,
                     (this->fe.component_mask(this->pressure)));
      }

      VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     0, //bottom face: fixed z-displacements
                     ZeroFunction<dim>(this->n_components),
                     constraints,
                     (this->fe.component_mask(this->x_displacement) |
                      this->fe.component_mask(this->y_displacement) |
                      this->fe.component_mask(this->z_displacement) ));

      if (this->parameters.load_type == "displacement")
        AssertThrow(false,
        ExcMessage("Displacement loading not defined for the current problem: "
                    + this->parameters.geom_type));
  }
};

template <int dim>
  class GrowthCubeConfinedUndrained
      : public GrowthBaseCube<dim>
{
public:
    GrowthCubeConfinedUndrained (const Parameters::AllParameters &parameters)
    : GrowthBaseCube<dim> (parameters)
  {}

  virtual ~GrowthCubeConfinedUndrained () {}

private:
  virtual void
  make_dirichlet_constraints(AffineConstraints<double> &constraints)
  {

      VectorTools::interpolate_boundary_values
                        (this->dof_handler_ref,
                         0, //bottom face: fixed z-displacements
                         ZeroFunction<dim>(this->n_components),
                         constraints,
                         (this->fe.component_mask(this->x_displacement) |
                          this->fe.component_mask(this->y_displacement) |
                          this->fe.component_mask(this->z_displacement) ));

      if (this->parameters.load_type == "displacement")
        AssertThrow(false,
          ExcMessage("Displacement loading not defined for the current problem: "
                      + this->parameters.geom_type));
  }
};


template <int dim>
  class GrowthCubeUnconfinedDrained
      : public GrowthBaseCube<dim>
{
public:
    GrowthCubeUnconfinedDrained (const Parameters::AllParameters &parameters)
    : GrowthBaseCube<dim> (parameters)
  {}

  virtual ~GrowthCubeUnconfinedDrained () {}

private:
  virtual void
  make_dirichlet_constraints(AffineConstraints<double> &constraints)
  {
      if (this->time.get_timestep()<2) //Dirichlet BC on pressure nodes
      {
          VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     0,
                     ConstantFunction<dim>(this->parameters.drained_pressure,
                                           this->n_components),
                     constraints,
                     (this->fe.component_mask(this->pressure)));
      }
      else
      {
          VectorTools::interpolate_boundary_values
                     (this->dof_handler_ref,
                      0,
                      ZeroFunction<dim>(this->n_components),
                      constraints,
                      (this->fe.component_mask(this->pressure)));
      }

      // Fully-fix a node at the center of the cube
      Point<dim> fix_node(0.5*this->parameters.scale,
                          0.5*this->parameters.scale,
                          0.5*this->parameters.scale);
      typename DoFHandler<dim>::active_cell_iterator
      cell = this->dof_handler_ref.begin_active(),
      endc = this->dof_handler_ref.end();
      for (; cell != endc; ++cell)
        for (unsigned int node=0; node<GeometryInfo<dim>::vertices_per_cell; ++node)
        {
            if ((abs(cell->vertex(node)[0]-fix_node[0])
                    <(1.0e-6*this->parameters.scale)) &&
                (abs(cell->vertex(node)[1]-fix_node[1])
                    <(1.0e-6*this->parameters.scale)) &&
                (abs(cell->vertex(node)[2]-fix_node[2])
                    <(1.0e-6*this->parameters.scale))    )
            {
                constraints.add_line(cell->vertex_dof_index(node, 0));
                constraints.add_line(cell->vertex_dof_index(node, 1));
                constraints.add_line(cell->vertex_dof_index(node, 2));

            }
        }

      if (this->parameters.load_type == "displacement")
      AssertThrow(false,
        ExcMessage("Displacement loading not defined for the current problem: "
          + this->parameters.geom_type));
  }
};

template <int dim>
  class GrowthCubeUnconfinedUndrained
      : public GrowthBaseCube<dim>
{
public:
    GrowthCubeUnconfinedUndrained (const Parameters::AllParameters &parameters)
    : GrowthBaseCube<dim> (parameters)
  {}

  virtual ~GrowthCubeUnconfinedUndrained () {}

private:
  virtual void
  make_dirichlet_constraints(AffineConstraints<double> &constraints)
  {
      // Fully-fix a node at the center of the cube
      Point<dim> fix_node(0.5*this->parameters.scale,
                          0.5*this->parameters.scale,
                          0.5*this->parameters.scale);
      typename DoFHandler<dim>::active_cell_iterator
      cell = this->dof_handler_ref.begin_active(),
      endc = this->dof_handler_ref.end();
      for (; cell != endc; ++cell)
        for (unsigned int node=0; node<GeometryInfo<dim>::vertices_per_cell; ++node)
        {
            if ((abs(cell->vertex(node)[0]-fix_node[0])
                      <(1.0e-6*this->parameters.scale)) &&
                (abs(cell->vertex(node)[1]-fix_node[1])
                      <(1.0e-6*this->parameters.scale)) &&
                (abs(cell->vertex(node)[2]-fix_node[2])
                      <(1.0e-6*this->parameters.scale))    )
            {
                constraints.add_line(cell->vertex_dof_index(node, 0));
                constraints.add_line(cell->vertex_dof_index(node, 1));
                constraints.add_line(cell->vertex_dof_index(node, 2));

            }
        }
  }
};


// @sect3{Axolotl limb examples}
// We group the definition of the geometry, boundary and loading conditions specific to
// the examples related to axolotl joint morphogenesis into specific classes.

// This function returns the great circle distance between two points
double GreatCircleDistance (const double joint_r, //sphere radius of geometry representing joint
                            const double theta_1, //Min azimuth angle of center of loading position
                            const double phi_1,   //Min polar angle of center of loading position
                            const double theta_2, //Max azimuth angle of center of loading position
                            const double phi_2)   //Min polar angle of center of loading position
  {
      // Here computations are done in spherical coordinates
      const double theta_12 = theta_2 - theta_1;
      const double X = std::sqrt((std::sin(phi_1)*std::cos(phi_1)
                                   -std::cos(phi_2)*std::sin(phi_2)*std::cos(theta_12))
                                  *(std::sin(phi_1)*std::cos(phi_1)
                                   -std::cos(phi_2)*std::sin(phi_2)*std::cos(theta_12))
                                 + (std::sin(phi_2)*std::sin(theta_12))
                                   *(std::sin(phi_2)*std::sin(theta_12)));
      const double Y = std::cos(phi_1)*std::cos(phi_2)
                        + std::sin(phi_1)*std::sin(phi_2)*std::cos(theta_12);
      const double sigma_12 = std::atan2(X,Y);
      const double distance = joint_r * std::abs(sigma_12);

      return distance;
  }

// This function returns the azimuth and polar angles of a point at a given distance
// along the great circle distance between two points.
Point<2> PointOnGreatCircle ( const double dist_point, //distance og the point
                              const double joint_r,  //sphere radius of geometry representing joint
                              const double theta_1,  //Min azimuth angle of center of loading position
                              const double phi_1_in,    //Min polar angle of center of loading position
                              const double theta_2,  //Max azimuth angle of center of loading position
                              const double phi_2_in)    //Min polar angle of center of loading position
  {
      
      // Convert polar angle to latitude because calculations are done in geographic coordinates.
      // Longitude = azimuth, no coversion needed.
      // (see Wikipedia webpage for great-circle navigation)
      const double phi_1 = 0.5*(numbers::PI) - phi_1_in;
      const double phi_2 = 0.5*(numbers::PI) - phi_2_in;
      
      // Calculations in geographic coordinates
      const double theta_12 = theta_2 - theta_1;
      
      const double Xa = std::cos(phi_2)*std::sin(theta_12);
      const double Ya = std::cos(phi_1)*std::sin(phi_2)
                        -std::sin(phi_1)*std::cos(phi_2)*std::cos(theta_12);
      const double alpha_1 = std::atan2(Xa,Ya);
         
      const double Xb = std::sin(alpha_1)*std::cos(phi_1);
      const double Yb = std::sqrt((std::cos(alpha_1))*(std::cos(alpha_1))
                                 +(std::sin(alpha_1))*(std::sin(alpha_1))
                                  *(std::sin(phi_1))*(std::sin(phi_1)));
      const double alpha_0 = std::atan2(Xb,Yb);
         
      double sigma_01;
      if ((phi_1 == 0.0) && (alpha_1 == 0.5*(numbers::PI)))
         sigma_01 = 0;
      else
      {
          const double Xc = std::tan(phi_1);
          const double Yc = std::cos(alpha_1);
          sigma_01 = std::atan2(Xc,Yc);
      }

      const double Xd = std::sin(alpha_0)*std::sin(sigma_01);
      const double Yd = std::cos(sigma_01);
      const double theta_01 = std::atan2(Xd,Yd);
         
      const double theta_0 = theta_1 - theta_01;
      const double sigma_p = sigma_01 + dist_point/joint_r;
         
      
      const double Xe = std::cos(alpha_0)*std::sin(sigma_p);
      const double Ye = std::sqrt((std::cos(sigma_p))*(std::cos(sigma_p))
                                  + (std::sin(alpha_0))*(std::sin(alpha_0))
                                  *(std::sin(sigma_p))*(std::sin(sigma_p)));

      const double phi_point = std::atan2(Xe,Ye);
      
      const double Xf = std::sin(alpha_0)*std::sin(sigma_p);
      const double Yf = std::cos(sigma_p);
      const double theta_point = std::atan2(Xf,Yf) + theta_0;
      
      // Convert latitude back to polar angle
      const double phi_point_out = 0.5*(numbers::PI) - phi_point;
      
      const Point<2> point_angles(theta_point,phi_point_out);
      return point_angles;
  }


// This class computes the distribution of loading on the load surface
// The central point has maximum value, the edges have zero value and
// we use a sinusoidal distribution for the rest
template <int dim>
class JointLoadingPattern : public Functions::Spherical<dim>
{
public:
  JointLoadingPattern(const double theta_load,  //Azimuth angle of center of loading position
                      const double phi_load,    //Polar angle of center of loading position
                      const double r_load,      //maximum radius of load contact surface
                      const double joint_r)     //sphere radius of geometry representing joint
  : Functions::Spherical<dim>(),
   theta_load(theta_load),
   phi_load(phi_load),
   r_load(r_load),
   joint_r(joint_r)
{}

private:
  virtual double svalue(const std::array<double,dim> &sp,
                        const unsigned int /*component*/) const;
  const double theta_load;
  const double phi_load;
  const double r_load;
  const double joint_r;
};

template <int dim>
double JointLoadingPattern<dim>::svalue(const std::array<double,dim> &sp,
                                        const unsigned int /*component*/) const
{
   double val = 0.0;
   const double &theta_p = sp[1]; //Azimuthal angle
   const double &phi_p = sp[2];   //Polar angle

   const double dist = joint_r *
                       std::acos( (std::sin(phi_p))*(std::sin(phi_load))*
                                  (std::cos(theta_p - theta_load)) +
                                  (std::cos(phi_p))*(std::cos(phi_load))  );

  if (std::abs(dist) <= r_load)
    val = std::cos(dist*(numbers::PI)/(2*r_load));

  Assert (dealii::numbers::is_finite(val), ExcInternalError());
  return val;
}

//@sect4{Base geometry of idealized humerus for biomechanical tissue-level model.}
template <int dim>
  class IdealisedHumerusBase
      : public Solid<dim>
{
public:
    IdealisedHumerusBase (const Parameters::AllParameters &parameters)
    : Solid<dim> (parameters)
  {}

  virtual ~IdealisedHumerusBase () {}

  virtual Tensor<1,dim>
  get_neumann_traction (const types::boundary_id &boundary_id,
                        const Point<dim>         &pt,
                        const Tensor<1,dim>      &N) const
  {
    Tensor<1,dim> load_vector;

    if (this->parameters.load_type == "pressure")
    {
      if ( (boundary_id == 1) || (boundary_id == 2) )
      {
           const double current_time = this->time.get_current();
           const double end_load_time = (this->time.get_delta_t())*
                                  (this->parameters.num_no_load_time_steps);
           const double final_load_time = (this->time.get_end()) - end_load_time;

           if (current_time <= final_load_time)
           {
             // Sinusoidal change between min and max positions of
             // loading trajectory. Loading trajectory computed as shortest
             // dist (great circle) between min and max positions.
               
                // RADIUS
                // Min and max polar angles of center of loading position (radius) in radians
                const double radius_phi_min_rad = (this->parameters.radius_phi_min)
                                                  *(numbers::PI)/180.;
                const double radius_phi_max_rad = (this->parameters.radius_phi_max)
                                                  *(numbers::PI)/180.;

                // Min and max azimuthal angles of center of loading position (radius) in radians
                const double radius_theta_min_rad = (this->parameters.radius_theta_min)
                                                    *(numbers::PI)/180.;
                const double radius_theta_max_rad = (this->parameters.radius_theta_max)
                                                    *(numbers::PI)/180.;

                // ULNA
                // Min and max polar angles of center of loading position (ulna) in radians
                const double ulna_phi_min_rad = (this->parameters.ulna_phi_min)
                                             *(numbers::PI)/180.;
                const double ulna_phi_max_rad = (this->parameters.ulna_phi_max)
                                             *(numbers::PI)/180.;

                // Min and max azimuthal angles of center of loading position (radius) in radians
                const double ulna_theta_min_rad = (this->parameters.ulna_theta_min)
                                               *(numbers::PI)/180.;
                const double ulna_theta_max_rad = (this->parameters.ulna_theta_max)
                                               *(numbers::PI)/180.;

                // Compute maximum distance (grand circle distance)
                // between max and min loading positions
                const double max_distance_radius =
                         GreatCircleDistance(this->parameters.joint_radius,
                         radius_theta_min_rad, radius_phi_min_rad,
                         radius_theta_max_rad, radius_phi_max_rad);

                const double max_distance_ulna =
                         GreatCircleDistance(this->parameters.joint_radius,
                         ulna_theta_min_rad, ulna_phi_min_rad,
                         ulna_theta_max_rad, ulna_phi_max_rad);

                const unsigned int num_cycles = this->parameters.num_cycles;
               
               // Compute current distance on great arc circle
               const double dist_points_radius = max_distance_radius
                  *(1.0 - std::sin((numbers::PI)
                  *(2.0*num_cycles*current_time/final_load_time + 0.5)))/2.0;

               const double dist_points_ulna = max_distance_ulna
                   *(1.0 - std::sin((numbers::PI)
                   *(2.0*num_cycles*current_time/final_load_time + 0.5)))/2.0;

               // (theta_point,phi_point);
               const Point<2> current_radius_load =
                    PointOnGreatCircle(dist_points_radius,
                                       this->parameters.joint_radius,
                                       radius_theta_min_rad,
                                       radius_phi_min_rad,
                                       radius_theta_max_rad,
                                       radius_phi_max_rad);

               const Point<2> current_ulna_load =
                    PointOnGreatCircle(dist_points_ulna,
                                       this->parameters.joint_radius,
                                       ulna_theta_min_rad,
                                       ulna_phi_min_rad,
                                       ulna_theta_max_rad,
                                       ulna_phi_max_rad);

                const JointLoadingPattern<dim>
                radius_load_spatial_distribution( current_radius_load[0],
                                                  current_radius_load[1],
                                                  this->parameters.radius_area_r,
                                                  this->parameters.joint_radius  );
                const JointLoadingPattern<dim>
                ulna_load_spatial_distribution( current_ulna_load[0],
                                                current_ulna_load[1],
                                                this->parameters.ulna_area_r,
                                                this->parameters.joint_radius  );
               
             // Load intensity reduced according to input factor.
             // Sinusoidal change between max value at start position of loading trajectory,
             // min value at max position, and max value at end position.
               
               
                const double max_load_value = this->parameters.load;
                const double min_load_value = max_load_value*(1.0 - this->parameters.load_reduction);
               
                double current_load_value = min_load_value + (max_load_value - min_load_value)*
               (1.0 + std::cos((numbers::PI)*(2.0*num_cycles*current_time/final_load_time)))/2.0;
               
                // Compute load vector for given position.
                load_vector = ( radius_load_spatial_distribution.value({pt[0],pt[1],pt[2]})
                                + ulna_load_spatial_distribution.value({pt[0],pt[1],pt[2]}) )
                              * current_load_value * N;
         }
      }
    }
    return load_vector;
  }

private:
  virtual void
  make_grid()
  {
      double radius = this->parameters.joint_radius;
      double cylinder_height = (this->parameters.joint_length)
                                -(this->parameters.joint_radius);
      Triangulation<dim>  tria_cylinder;
      GridGenerator::cylinder(tria_cylinder,
                              radius,
                              0.5*cylinder_height);
      //Create a cylinder around the x-axis. The cylinder extends from
      //x=-"half_length" to x=+"half_length" and its projection into the
      //yz-plane is a circle of radius "radius".
      //The boundaries are colored according to the following scheme:
      //0 for the hull of the cylinder,
      //1 for the left hand face and
      //2 for the right hand face.

     //Rotate cylinder so that it is aligned with the z-axis
     const double rot_angle = 3.0*(numbers::PI)/2.0;
     GridTools::rotate(rot_angle, 1, tria_cylinder);
     const Tensor<1,dim> shift_cylinder({0.0, 0.0, -0.5*cylinder_height});
     GridTools::shift(shift_cylinder, tria_cylinder);

     // Hull of cylinder = drained boundary        --> 0
     // Left hand face is now bottom face = fixed  --> 1
     // Right hand face is now top face = load     --> 2

     const Point< dim > center(0.0,0.0,0.0);
     Triangulation<dim>  tria_half_sphere;
     GridGenerator::half_hyper_ball(tria_half_sphere,
                                    center,
                                    radius            );
     //A half hyper-ball around center, which contains 6 in 3d.
     //The cut plane is perpendicular to the x-axis.
     //Boundary indicators are 0 for the curved boundary and 1 for the cut plane.
     //The manifold id for the curved boundary is set to zero, and
     //a SphericalManifold is attached to it.

     //Rotate half-sphere so that it is perpendicular to the z-axis
     GridTools::rotate(rot_angle, 1, tria_half_sphere);

     //Merge the two meshes
     GridGenerator::merge_triangulations(tria_cylinder,
                                         tria_half_sphere,
                                         this->triangulation);

    // Assign boundary IDs
    for (auto cell : this->triangulation.active_cell_iterators())
       for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
          if (cell->face(face)->at_boundary() == true  &&
              std::abs(cell->face(face)->center()[2]+cylinder_height)<1.0e-6)
                      cell->face(face)->set_boundary_id(0);  //Bottom face

          else if (cell->face(face)->at_boundary() == true  &&
                   cell->face(face)->center()[2] < 0.0      &&
                   cell->face(face)->center()[2] > -1.0*cylinder_height )
                      cell->face(face)->set_boundary_id(1); //Hull of cylinder

          else if (cell->face(face)->at_boundary() == true  &&
                   cell->face(face)->center()[2] > 0.0         )
                      cell->face(face)->set_boundary_id(2); //Spherical surface

    //Set manifolds
    const types::manifold_id  sphere_id = 0;
    const types::manifold_id  inner_id = 1;
    const types::manifold_id  cylinder_id = 2;

    const SphericalManifold<dim> spherical_manifold(center);
    TransfiniteInterpolationManifold<dim> inner_manifold;
    const CylindricalManifold<dim> cylindrical_manifold(2);

    //Assign manifold IDs
    this->triangulation.set_all_manifold_ids(cylinder_id);
    for (auto cell : this->triangulation.active_cell_iterators())
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        if (std::abs(cell->face(face)->center()[0])<1.0e-6 &&
            std::abs(cell->face(face)->center()[1])<1.0e-6 &&
            cell->center()[2]<0.0                             )
              cell->set_all_manifold_ids(numbers::flat_manifold_id);
    for (auto cell : this->triangulation.active_cell_iterators())
      if (cell->center()[2]>0.0)
              cell->set_all_manifold_ids(inner_id);
    this->triangulation.set_all_manifold_ids_on_boundary(2,sphere_id);

    //Set manifold
    this->triangulation.set_manifold (cylinder_id, cylindrical_manifold);
    this->triangulation.set_manifold (sphere_id, spherical_manifold);
    inner_manifold.initialize(this->triangulation);
    this->triangulation.set_manifold (inner_id, inner_manifold);

     //Refine mesh
     this->triangulation.refine_global(1);
     inner_manifold.initialize(this->triangulation);
     if (this->parameters.global_refinement > 1)
      this->triangulation.refine_global((this->parameters.global_refinement)-1);

     //Scale geometry
     GridTools::scale(this->parameters.scale, this->triangulation);
  }

  virtual void
  define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices)
  {
    tracked_vertices[0][0] = 0.0*this->parameters.scale;
    tracked_vertices[0][1] = 0.0*this->parameters.scale;
    tracked_vertices[0][2] = (this->parameters.joint_radius)
                              *this->parameters.scale;
    tracked_vertices[1][0] = 0.0*this->parameters.scale;
    tracked_vertices[1][1] = 0.0*this->parameters.scale;
    tracked_vertices[1][2] = ((this->parameters.joint_radius)
                              -(this->parameters.joint_length))
                              *this->parameters.scale;
  }

  virtual double
  get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                             const Point<dim>         &pt) const
  {
      //Silence compiler warnings
      (void)pt;
      (void)boundary_id;
      return 0.0;
  }

  virtual std::pair<types::boundary_id,types::boundary_id>
  get_reaction_boundary_id_for_output() const
  {
      return std::make_pair(1,2);
  }

  virtual std::pair<types::boundary_id,types::boundary_id>
  get_drained_boundary_id_for_output() const
  {
      return std::make_pair(1,2);
  }

  virtual std::pair<double, FEValuesExtractors::Scalar>
  get_dirichlet_load(const types::boundary_id &boundary_id) const
  {
      double displ_incr = 0;
      FEValuesExtractors::Scalar direction;
      (void)boundary_id;
      return std::make_pair(displ_incr,direction);
  }
};

//@sect4{Loaded surface undrained, rest is drained} NOT WORKING PROPERLY!!!
template <int dim>
  class IdealisedHumerusPartiallyDrained
      : public IdealisedHumerusBase<dim>
{
public:
    IdealisedHumerusPartiallyDrained (const Parameters::AllParameters &parameters)
    : IdealisedHumerusBase<dim> (parameters)
  {}

  virtual ~IdealisedHumerusPartiallyDrained () {}
    
private:
  virtual void
  make_dirichlet_constraints(AffineConstraints<double> &constraints)
  {
       // Dirichlet BCs on displacements
       if (this->parameters.load_type == "displacement")
       AssertThrow(false,
         ExcMessage("Displacement loading not defined for the current problem: "
                     + this->parameters.geom_type));
      
       // Fix vertical displ of bottom surface
       VectorTools::interpolate_boundary_values
                        (this->dof_handler_ref,
                         0,
                         ZeroFunction<dim>(this->n_components),
                         constraints,
                         this->fe.component_mask(this->z_displacement) );

       // Fix x and y displ of central node in bottom surface
       Point<2> fix_node(0.0, 0.0);
       Tensor<1,dim> N;
       N[0]=1.0;
       N[1]=0.0;
       N[2]=0.0;
      
       std::vector<unsigned int> face_dof_indices (this->fe.dofs_per_face);
      
       typename DoFHandler<dim>::active_cell_iterator
       cell = this->dof_handler_ref.begin_active(),
       endc = this->dof_handler_ref.end();
       for (; cell != endc; ++cell)
         for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
         {
             if ( cell->face(face)->at_boundary() == true  &&
                  cell->face(face)->boundary_id() == 0        )
             {
                 for (unsigned int node=0; node<GeometryInfo<dim>::vertices_per_face; ++node)
                 {
                     if ( abs(cell->face(face)->vertex(node)[0]-fix_node[0])
                           < (1.0e-6*this->parameters.scale) )
                        constraints.add_line(cell->vertex_dof_index(node, 0));

                     if ( abs(cell->face(face)->vertex(node)[1]-fix_node[1])
                           < (1.0e-6*this->parameters.scale) )
                        constraints.add_line(cell->vertex_dof_index(node, 1));
                 }
             }
             
             // Dirichlet BCs on pressure
             // For lateral and top surfaces, define constraints based on Neumann load
             if ( cell->face(face)->at_boundary() == true  &&
                  (cell->face(face)->boundary_id() == 1 ||
                   cell->face(face)->boundary_id() == 2   )   )
             {
                 // Check value of Neumann load.
                 Point<dim> pt;
                 pt[0] = cell->face(face)->center()[0];
                 pt[1] = cell->face(face)->center()[1];
                 pt[2] = cell->face(face)->center()[2];
                 
                 Tensor<1,dim> load =
                   this->get_neumann_traction(cell->face(face)->boundary_id(), pt, N);
                 
                 // If no load at central point of this face, then apply
                 // Dirichlet constraint on pressure dof of all nodes in face
                 if ( abs(load[0]) < 1.0e-10 )
                 {
                     // Get all dofs in face
                     cell->face(face)->get_dof_indices(face_dof_indices);
                     
                     //Loop over dofs. If it is a pressure dof, add constraint.
                     for (unsigned int i = 0; i<face_dof_indices.size(); ++i)
                        if (this->fe.face_system_to_base_index(i).first.first
                             == this->p_fluid_block)
                            constraints.add_line (face_dof_indices[i]);
                 }
             }
         }
          
       // Dirichlet BCs on pressure
       // Free flow (pressure = 0) on bottom surface
       VectorTools::interpolate_boundary_values
                (this->dof_handler_ref,
                 0,
                 ZeroFunction<dim>(this->n_components),
                 constraints,
                 (this->fe.component_mask(this->pressure)));
  }
};
    
//@sect4{All boundaries drained, including the loaded surfaces}
template <int dim>
class IdealisedHumerusFullyDrained
  : public IdealisedHumerusBase<dim>
{
public:
  IdealisedHumerusFullyDrained (const Parameters::AllParameters &parameters)
  : IdealisedHumerusBase<dim> (parameters)
  {}

virtual ~IdealisedHumerusFullyDrained () {}

private:
  virtual void
  make_dirichlet_constraints(AffineConstraints<double> &constraints)
  {
      
      // Dirichlet BCs on displacements
      if (this->parameters.load_type == "displacement")
      AssertThrow(false,
         ExcMessage("Displacement loading not defined for the current problem: "
                     + this->parameters.geom_type));
      
      // Fix vertical displ of bottom surface
      VectorTools::interpolate_boundary_values
                        (this->dof_handler_ref,
                         0,
                         ZeroFunction<dim>(this->n_components),
                         constraints,
                         this->fe.component_mask(this->z_displacement) );

       // Fix x and y displ of central node in bottom surface
       Point<2> fix_node(0.0, 0.0);

       typename DoFHandler<dim>::active_cell_iterator
       cell = this->dof_handler_ref.begin_active(),
       endc = this->dof_handler_ref.end();
       for (; cell != endc; ++cell)
         for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
           if ( cell->face(face)->at_boundary() == true  &&
                cell->face(face)->boundary_id() == 0        )
             for (unsigned int node=0; node<GeometryInfo<dim>::vertices_per_face; ++node)
             {
                 if ( abs(cell->face(face)->vertex(node)[0]-fix_node[0])
                       < (1.0e-6*this->parameters.scale) )
                    constraints.add_line(cell->vertex_dof_index(node, 0));

                 if ( abs(cell->face(face)->vertex(node)[1]-fix_node[1])
                       < (1.0e-6*this->parameters.scale) )
                    constraints.add_line(cell->vertex_dof_index(node, 1));
             }

       // Dirichlet BCs on pressure
       // Free flow (pressure = 0) on all surfaces
       VectorTools::interpolate_boundary_values
                (this->dof_handler_ref,
                 0,
                 ZeroFunction<dim>(this->n_components),
                 constraints,
                 (this->fe.component_mask(this->pressure)));

       VectorTools::interpolate_boundary_values
                (this->dof_handler_ref,
                 1,
                 ZeroFunction<dim>(this->n_components),
                 constraints,
                 (this->fe.component_mask(this->pressure)));
      
        VectorTools::interpolate_boundary_values
                 (this->dof_handler_ref,
                  2,
                  ZeroFunction<dim>(this->n_components),
                  constraints,
                  (this->fe.component_mask(this->pressure)));
  }
};

//@sect4{Cylindrical and spherical surfaces undrained, bottom surface drained}
template <int dim>
class IdealisedHumerusLateralUndrained
  : public IdealisedHumerusBase<dim>
{
public:
  IdealisedHumerusLateralUndrained (const Parameters::AllParameters &parameters)
  : IdealisedHumerusBase<dim> (parameters)
  {}

virtual ~IdealisedHumerusLateralUndrained () {}

private:
virtual void
make_dirichlet_constraints(AffineConstraints<double> &constraints)
{
  
  // Dirichlet BCs on displacements
  if (this->parameters.load_type == "displacement")
  AssertThrow(false,
     ExcMessage("Displacement loading not defined for the current problem: "
                 + this->parameters.geom_type));
  
  // Fix vertical displ of bottom surface
  VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     0,
                     ZeroFunction<dim>(this->n_components),
                     constraints,
                     this->fe.component_mask(this->z_displacement) );

   // Fix x and y displ of central node in bottom surface
   Point<2> fix_node(0.0, 0.0);

   typename DoFHandler<dim>::active_cell_iterator
   cell = this->dof_handler_ref.begin_active(),
   endc = this->dof_handler_ref.end();
   for (; cell != endc; ++cell)
     for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
       if ( cell->face(face)->at_boundary() == true  &&
            cell->face(face)->boundary_id() == 0        )
         for (unsigned int node=0; node<GeometryInfo<dim>::vertices_per_face; ++node)
         {
             if ( abs(cell->face(face)->vertex(node)[0]-fix_node[0])
                   < (1.0e-6*this->parameters.scale) )
                constraints.add_line(cell->vertex_dof_index(node, 0));

             if ( abs(cell->face(face)->vertex(node)[1]-fix_node[1])
                   < (1.0e-6*this->parameters.scale) )
                constraints.add_line(cell->vertex_dof_index(node, 1));
         }

   // Dirichlet BCs on pressure
   // Free flow (pressure = 0) only on bottom surface
   VectorTools::interpolate_boundary_values
            (this->dof_handler_ref,
             0,
             ZeroFunction<dim>(this->n_components),
             constraints,
             (this->fe.component_mask(this->pressure)));
  }
};

//@sect4{Base geometry of humerus meshed externally from an stl file for biomechanical tissue-level model.}
template <int dim>
  class ExternalMeshHumerusBase
      : public Solid<dim>
{
public:
    ExternalMeshHumerusBase (const Parameters::AllParameters &parameters)
    : Solid<dim> (parameters)
  {}

  virtual ~ExternalMeshHumerusBase () {}

  virtual Tensor<1,dim>
  get_neumann_traction (const types::boundary_id &boundary_id,
                        const Point<dim>         &pt,
                        const Tensor<1,dim>      &N) const
  {
    Tensor<1,dim> load_vector;

    if (this->parameters.load_type == "pressure")
    {
      if ( (boundary_id == 1) || (boundary_id == 2) )
      {
           const double current_time = this->time.get_current();
           const double end_load_time = (this->time.get_delta_t())*
                                  (this->parameters.num_no_load_time_steps);
           const double final_load_time = (this->time.get_end()) - end_load_time;

           if (current_time <= final_load_time)
           {
             // Sinusoidal change between min and max positions of
             // loading trajectory. Loading trajectory computed as shortest
             // dist (great circle) between min and max positions.
               
                // RADIUS
                // Min and max polar angles of center of loading position (radius) in radians
                const double radius_phi_min_rad = (this->parameters.radius_phi_min)
                                                  *(numbers::PI)/180.;
                const double radius_phi_max_rad = (this->parameters.radius_phi_max)
                                                  *(numbers::PI)/180.;

                // Min and max azimuthal angles of center of loading position (radius) in radians
                const double radius_theta_min_rad = (this->parameters.radius_theta_min)
                                                    *(numbers::PI)/180.;
                const double radius_theta_max_rad = (this->parameters.radius_theta_max)
                                                    *(numbers::PI)/180.;

                // ULNA
                // Min and max polar angles of center of loading position (ulna) in radians
                const double ulna_phi_min_rad = (this->parameters.ulna_phi_min)
                                             *(numbers::PI)/180.;
                const double ulna_phi_max_rad = (this->parameters.ulna_phi_max)
                                             *(numbers::PI)/180.;

                // Min and max azimuthal angles of center of loading position (radius) in radians
                const double ulna_theta_min_rad = (this->parameters.ulna_theta_min)
                                               *(numbers::PI)/180.;
                const double ulna_theta_max_rad = (this->parameters.ulna_theta_max)
                                               *(numbers::PI)/180.;

                // Compute maximum distance (grand circle distance)
                // between max and min loading positions
                const double max_distance_radius =
                         GreatCircleDistance(this->parameters.joint_radius,
                         radius_theta_min_rad, radius_phi_min_rad,
                         radius_theta_max_rad, radius_phi_max_rad);

                const double max_distance_ulna =
                         GreatCircleDistance(this->parameters.joint_radius,
                         ulna_theta_min_rad, ulna_phi_min_rad,
                         ulna_theta_max_rad, ulna_phi_max_rad);

                const unsigned int num_cycles = this->parameters.num_cycles;
               
               // Compute current distance on great arc circle
               const double dist_points_radius = max_distance_radius
                  *(1.0 - std::sin((numbers::PI)
                  *(2.0*num_cycles*current_time/final_load_time + 0.5)))/2.0;

               const double dist_points_ulna = max_distance_ulna
                   *(1.0 - std::sin((numbers::PI)
                   *(2.0*num_cycles*current_time/final_load_time + 0.5)))/2.0;

               // (theta_point,phi_point);
               const Point<2> current_radius_load =
                    PointOnGreatCircle(dist_points_radius,
                                       this->parameters.joint_radius,
                                       radius_theta_min_rad,
                                       radius_phi_min_rad,
                                       radius_theta_max_rad,
                                       radius_phi_max_rad);

               const Point<2> current_ulna_load =
                    PointOnGreatCircle(dist_points_ulna,
                                       this->parameters.joint_radius,
                                       ulna_theta_min_rad,
                                       ulna_phi_min_rad,
                                       ulna_theta_max_rad,
                                       ulna_phi_max_rad);

                const JointLoadingPattern<dim>
                radius_load_spatial_distribution( current_radius_load[0],
                                                  current_radius_load[1],
                                                  this->parameters.radius_area_r,
                                                  this->parameters.joint_radius  );
                const JointLoadingPattern<dim>
                ulna_load_spatial_distribution( current_ulna_load[0],
                                                current_ulna_load[1],
                                                this->parameters.ulna_area_r,
                                                this->parameters.joint_radius  );
               
             // Load intensity reduced according to input factor.
             // Sinusoidal change between max value at start position of loading trajectory,
             // min value at max position, and max value at end position.
               
               
                const double max_load_value = this->parameters.load;
                const double min_load_value = max_load_value*(1.0 - this->parameters.load_reduction);
               
                double current_load_value = min_load_value + (max_load_value - min_load_value)*
               (1.0 + std::cos((numbers::PI)*(2.0*num_cycles*current_time/final_load_time)))/2.0;
               
                // Compute load vector for given position.
                load_vector = ( radius_load_spatial_distribution.value({pt[0],pt[1],pt[2]})
                                + ulna_load_spatial_distribution.value({pt[0],pt[1],pt[2]}) )
                              * current_load_value * N;
         }
      }
    }
    return load_vector;
  }

private:
  virtual void
  make_grid()
  {
      //Read external mesh
      GridIn<dim> gridin;
      gridin.attach_triangulation(this->triangulation);
      std::ifstream input_file("humerus_mesh.inp");
      gridin.read_abaqus(input_file);
        
        
      //double radius = this->parameters.joint_radius;
      double cylinder_height = (this->parameters.joint_length)
                                -(this->parameters.joint_radius);

      // Assign boundary IDs
      for (auto cell : this->triangulation.active_cell_iterators())
          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            if (cell->face(face)->at_boundary() == true  &&
                std::abs(cell->face(face)->center()[2]+cylinder_height)<1.0e-6)
                      cell->face(face)->set_boundary_id(0);  //Bottom face

            else if (cell->face(face)->at_boundary() == true  &&
                     cell->face(face)->center()[2] < 0.0      &&
                     cell->face(face)->center()[2] > -1.0*cylinder_height )
                        cell->face(face)->set_boundary_id(1); //Surface of shaft

            else if (cell->face(face)->at_boundary() == true  &&
                     cell->face(face)->center()[2] > 0.0         )
                        cell->face(face)->set_boundary_id(2); //Head surface

     //Scale geometry
     GridTools::scale(this->parameters.scale, this->triangulation);
  }

  virtual void
  define_tracked_vertices(std::vector<Point<dim>> &tracked_vertices)
  {
    tracked_vertices[0][0] = 0.0*this->parameters.scale;
    tracked_vertices[0][1] = 0.0*this->parameters.scale;
    tracked_vertices[0][2] = 0.0*this->parameters.scale;
    tracked_vertices[1][0] = 0.0*this->parameters.scale;
    tracked_vertices[1][1] = 0.0*this->parameters.scale;
    tracked_vertices[1][2] = ((this->parameters.joint_radius)
                                -(this->parameters.joint_length))
                                *this->parameters.scale;
  }

  virtual double
  get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                             const Point<dim>         &pt) const
  {
      //Silence compiler warnings
      (void)pt;
      (void)boundary_id;
      return 0.0;
  }

  virtual std::pair<types::boundary_id,types::boundary_id>
  get_reaction_boundary_id_for_output() const
  {
      return std::make_pair(1,2);
  }

  virtual std::pair<types::boundary_id,types::boundary_id>
  get_drained_boundary_id_for_output() const
  {
      return std::make_pair(1,2);
  }

  virtual std::pair<double, FEValuesExtractors::Scalar>
  get_dirichlet_load(const types::boundary_id &boundary_id) const
  {
      double displ_incr = 0;
      FEValuesExtractors::Scalar direction;
      (void)boundary_id;
      return std::make_pair(displ_incr,direction);
  }
};

//@sect4{Loaded surface undrained, rest is drained} NOT WORKING PROPERLY!!!
template <int dim>
  class ExternalMeshHumerusPartiallyDrained
      : public ExternalMeshHumerusBase<dim>
{
public:
    ExternalMeshHumerusPartiallyDrained (const Parameters::AllParameters &parameters)
    : ExternalMeshHumerusBase<dim> (parameters)
  {}

  virtual ~ExternalMeshHumerusPartiallyDrained () {}
    
private:
  virtual void
  make_dirichlet_constraints(AffineConstraints<double> &constraints)
  {
       // Dirichlet BCs on displacements
       if (this->parameters.load_type == "displacement")
       AssertThrow(false,
         ExcMessage("Displacement loading not defined for the current problem: "
                     + this->parameters.geom_type));
      
       // Fix vertical displ of bottom surface
       VectorTools::interpolate_boundary_values
                        (this->dof_handler_ref,
                         0,
                         ZeroFunction<dim>(this->n_components),
                         constraints,
                         this->fe.component_mask(this->z_displacement) );

       // Fix x and y displ of central node in bottom surface
       Point<2> fix_node(0.0, 0.0);
       Tensor<1,dim> N;
       N[0]=1.0;
       N[1]=0.0;
       N[2]=0.0;
      
       std::vector<unsigned int> face_dof_indices (this->fe.dofs_per_face);
      
       typename DoFHandler<dim>::active_cell_iterator
       cell = this->dof_handler_ref.begin_active(),
       endc = this->dof_handler_ref.end();
       for (; cell != endc; ++cell)
         for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
         {
             if ( cell->face(face)->at_boundary() == true  &&
                  cell->face(face)->boundary_id() == 0        )
             {
                 for (unsigned int node=0; node<GeometryInfo<dim>::vertices_per_face; ++node)
                 {
                     if ( abs(cell->face(face)->vertex(node)[0]-fix_node[0])
                           < (1.0e-6*this->parameters.scale) )
                        constraints.add_line(cell->vertex_dof_index(node, 0));

                     if ( abs(cell->face(face)->vertex(node)[1]-fix_node[1])
                           < (1.0e-6*this->parameters.scale) )
                        constraints.add_line(cell->vertex_dof_index(node, 1));
                 }
             }
             
             // Dirichlet BCs on pressure
             // For lateral and top surfaces, define constraints based on Neumann load
             if ( cell->face(face)->at_boundary() == true  &&
                  (cell->face(face)->boundary_id() == 1 ||
                   cell->face(face)->boundary_id() == 2   )   )
             {
                 // Check value of Neumann load.
                 Point<dim> pt;
                 pt[0] = cell->face(face)->center()[0];
                 pt[1] = cell->face(face)->center()[1];
                 pt[2] = cell->face(face)->center()[2];
                 
                 Tensor<1,dim> load =
                   this->get_neumann_traction(cell->face(face)->boundary_id(), pt, N);
                 
                 // If no load at central point of this face, then apply
                 // Dirichlet constraint on pressure dof of all nodes in face
                 if ( abs(load[0]) < 1.0e-10 )
                 {
                     // Get all dofs in face
                     cell->face(face)->get_dof_indices(face_dof_indices);
                     
                     //Loop over dofs. If it is a pressure dof, add constraint.
                     for (unsigned int i = 0; i<face_dof_indices.size(); ++i)
                        if (this->fe.face_system_to_base_index(i).first.first
                             == this->p_fluid_block)
                            constraints.add_line (face_dof_indices[i]);
                 }
             }
         }
          
       // Dirichlet BCs on pressure
       // Free flow (pressure = 0) on bottom surface
       VectorTools::interpolate_boundary_values
                (this->dof_handler_ref,
                 0,
                 ZeroFunction<dim>(this->n_components),
                 constraints,
                 (this->fe.component_mask(this->pressure)));
  }
};
    
//@sect4{All boundaries drained, including the loaded surfaces}
template <int dim>
class ExternalMeshHumerusFullyDrained
  : public ExternalMeshHumerusBase<dim>
{
public:
    ExternalMeshHumerusFullyDrained (const Parameters::AllParameters &parameters)
  : ExternalMeshHumerusBase<dim> (parameters)
  {}

virtual ~ExternalMeshHumerusFullyDrained () {}

private:
  virtual void
  make_dirichlet_constraints(AffineConstraints<double> &constraints)
  {
      
      // Dirichlet BCs on displacements
      if (this->parameters.load_type == "displacement")
      AssertThrow(false,
         ExcMessage("Displacement loading not defined for the current problem: "
                     + this->parameters.geom_type));
      
      // Fix vertical displ of bottom surface
      VectorTools::interpolate_boundary_values
                        (this->dof_handler_ref,
                         0,
                         ZeroFunction<dim>(this->n_components),
                         constraints,
                         this->fe.component_mask(this->z_displacement) );

       // Fix x and y displ of central node in bottom surface
       Point<2> fix_node(0.0, 0.0);

       typename DoFHandler<dim>::active_cell_iterator
       cell = this->dof_handler_ref.begin_active(),
       endc = this->dof_handler_ref.end();
       for (; cell != endc; ++cell)
         for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
           if ( cell->face(face)->at_boundary() == true  &&
                cell->face(face)->boundary_id() == 0        )
             for (unsigned int node=0; node<GeometryInfo<dim>::vertices_per_face; ++node)
             {
                 if ( abs(cell->face(face)->vertex(node)[0]-fix_node[0])
                       < (1.0e-6*this->parameters.scale) )
                    constraints.add_line(cell->vertex_dof_index(node, 0));

                 if ( abs(cell->face(face)->vertex(node)[1]-fix_node[1])
                       < (1.0e-6*this->parameters.scale) )
                    constraints.add_line(cell->vertex_dof_index(node, 1));
             }

       // Dirichlet BCs on pressure
       // Free flow (pressure = 0) on all surfaces
       VectorTools::interpolate_boundary_values
                (this->dof_handler_ref,
                 0,
                 ZeroFunction<dim>(this->n_components),
                 constraints,
                 (this->fe.component_mask(this->pressure)));

       VectorTools::interpolate_boundary_values
                (this->dof_handler_ref,
                 1,
                 ZeroFunction<dim>(this->n_components),
                 constraints,
                 (this->fe.component_mask(this->pressure)));
      
        VectorTools::interpolate_boundary_values
                 (this->dof_handler_ref,
                  2,
                  ZeroFunction<dim>(this->n_components),
                  constraints,
                  (this->fe.component_mask(this->pressure)));
  }
};

//@sect4{Cylindrical and spherical surfaces undrained, bottom surface drained}
template <int dim>
class ExternalMeshHumerusLateralUndrained
  : public ExternalMeshHumerusBase<dim>
{
public:
    ExternalMeshHumerusLateralUndrained (const Parameters::AllParameters &parameters)
  : ExternalMeshHumerusBase<dim> (parameters)
  {}

virtual ~ExternalMeshHumerusLateralUndrained () {}

private:
virtual void
make_dirichlet_constraints(AffineConstraints<double> &constraints)
{
  
  // Dirichlet BCs on displacements
  if (this->parameters.load_type == "displacement")
  AssertThrow(false,
     ExcMessage("Displacement loading not defined for the current problem: "
                 + this->parameters.geom_type));
  
  // Fix vertical displ of bottom surface
  VectorTools::interpolate_boundary_values
                    (this->dof_handler_ref,
                     0,
                     ZeroFunction<dim>(this->n_components),
                     constraints,
                     this->fe.component_mask(this->z_displacement) );

   // Fix x and y displ of central node in bottom surface
   Point<2> fix_node(0.0, 0.0);

   typename DoFHandler<dim>::active_cell_iterator
   cell = this->dof_handler_ref.begin_active(),
   endc = this->dof_handler_ref.end();
   for (; cell != endc; ++cell)
     for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
       if ( cell->face(face)->at_boundary() == true  &&
            cell->face(face)->boundary_id() == 0        )
         for (unsigned int node=0; node<GeometryInfo<dim>::vertices_per_face; ++node)
         {
             if ( abs(cell->face(face)->vertex(node)[0]-fix_node[0])
                   < (1.0e-6*this->parameters.scale) )
                constraints.add_line(cell->vertex_dof_index(node, 0));

             if ( abs(cell->face(face)->vertex(node)[1]-fix_node[1])
                   < (1.0e-6*this->parameters.scale) )
                constraints.add_line(cell->vertex_dof_index(node, 1));
         }

   // Dirichlet BCs on pressure
   // Free flow (pressure = 0) only on bottom surface
   VectorTools::interpolate_boundary_values
            (this->dof_handler_ref,
             0,
             ZeroFunction<dim>(this->n_components),
             constraints,
             (this->fe.component_mask(this->pressure)));
  }
};
    
}


// @sect3{Main function}
// Lastly we provide the main driver function which is similar to the other tutorials.
int main (int argc, char *argv[])
{
using namespace dealii;
using namespace CompLimb;

const unsigned int n_tbb_processes = 1;
Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, n_tbb_processes);

try
  {
    Parameters::AllParameters parameters ("parameters.prm");
    if (parameters.geom_type == "growing_muffin")
    {
      GrowingMuffin<3> solid_3d(parameters);
      solid_3d.run();
    }
    else if (parameters.geom_type == "trapped_turtle")
    {
      TrappedTurtle<3> solid_3d(parameters);
      solid_3d.run();
    }
    else if (parameters.geom_type == "cube_growth_confined_drained")
    {
      GrowthCubeConfinedDrained<3> solid_3d(parameters);
      solid_3d.run();
    }
    else if (parameters.geom_type == "cube_growth_confined_undrained")
    {
      GrowthCubeConfinedUndrained<3> solid_3d(parameters);
      solid_3d.run();
    }
    else if (parameters.geom_type == "cube_growth_unconfined_drained")
    {
      GrowthCubeUnconfinedDrained<3> solid_3d(parameters);
      solid_3d.run();
    }
    else if (parameters.geom_type == "cube_growth_unconfined_undrained")
    {
      GrowthCubeUnconfinedUndrained<3> solid_3d(parameters);
      solid_3d.run();
    }
    else if (parameters.geom_type == "idealised_humerus_partially_drained")
    {
      IdealisedHumerusPartiallyDrained<3> solid_3d(parameters);
      solid_3d.run();
    }
    else if (parameters.geom_type == "idealised_humerus_fully_drained")
    {
      IdealisedHumerusFullyDrained<3> solid_3d(parameters);
      solid_3d.run();
    }
    else if (parameters.geom_type == "idealised_humerus_laterals_undrained")
    {
      IdealisedHumerusLateralUndrained<3> solid_3d(parameters);
      solid_3d.run();
    }
    else if (parameters.geom_type == "external_mesh_humerus_partially_drained")
    {
      ExternalMeshHumerusPartiallyDrained<3> solid_3d(parameters);
      solid_3d.run();
    }
    else if (parameters.geom_type == "external_mesh_humerus_fully_drained")
    {
       ExternalMeshHumerusFullyDrained<3> solid_3d(parameters);
      solid_3d.run();
    }
    else if (parameters.geom_type == "external_mesh_humerus_laterals_undrained")
    {
      ExternalMeshHumerusLateralUndrained<3> solid_3d(parameters);
      solid_3d.run();
    }
    else
    {
      AssertThrow(false,
                  ExcMessage("Problem type not defined. Current setting: "
                              + parameters.geom_type));
    }

  }
catch (std::exception &exc)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl << exc.what()
                  << std::endl << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    }
  }
catch (...)
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
  }
return 0;
}
