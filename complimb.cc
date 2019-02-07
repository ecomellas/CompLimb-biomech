/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2010 - 2019 by the deal.II authors and
 *                              Ester Comellas and Jean-Paul Pelteret
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
 *  Ester Comellas, Northeastern University, 2019
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
#include <deal.II/base/std_cxx11/shared_ptr.h>

#include <deal.II/differentiation/ad.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>
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
        unsigned int quad_order;

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

          prm.declare_entry("Quadrature order", "3",
                            Patterns::Integer(0),
                            "Gauss quadrature order");
        }
        prm.leave_subsection();
      }

      void FESystem::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Finite element system");
        {
          poly_degree_displ = prm.get_integer("Polynomial degree displ");
          poly_degree_pore = prm.get_integer("Polynomial degree pore");
          quad_order = prm.get_integer("Quadrature order");
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
        unsigned int num_cycle_sets;
        double       fluid_flow;
        double       drained_pressure;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
      };

      void Geometry::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Geometry");
        {
          prm.declare_entry("Geometry type", "Ehlers_tube_step_load",
                             Patterns::Selection("Ehlers_tube_step_load|Ehlers_tube_increase_load|Ehlers_cube_consolidation"
                                                 "|Franceschini_consolidation"
                                                 "|Budday_cube_tension_compression|Budday_cube_tension_compression_fully_fixed"
                                                 "|Budday_cube_shear_fully_fixed"
                                                 "|growing_muffin|trapped_turtle"
                                                 "|brain_growth_confined_drained|brain_growth_confined_undrained"
                                                 "|brain_growth_unconfined_drained|brain_growth_unconfined_undrained"),
                                "Type of geometry used. "
                                "For Ehlers validation examples see Ehlers and Eipper (1999). "
                                "For Franceschini brain consolidation see Franceschini et al. (2006)"
                                "For Budday brain examples see Budday et al. (2017)");

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

          prm.declare_entry("Number of cycle sets", "1",
                            Patterns::Integer(1,2),
                            "Number of times each set of 3 cycles is repeated, only for "
                            "Budday_cube_tension_compression and Budday_cube_tension_compression_fully_fixed. "
                            "Load value is doubled in second set, load rate is kept constant."
                            "Final time indicates end of second cycle set.");

          prm.declare_entry("Fluid flow value", "0.0",
                            Patterns::Double(),
                            "Prescribed fluid flow. Not implemented in any example yet.");

          prm.declare_entry("Drained pressure", "0.0",
                            Patterns::Double(),
                            "Increase of pressure value at drained boundary w.r.t the atmospheric pressure.");
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
          num_cycle_sets = prm.get_integer("Number of cycle sets");
          fluid_flow = prm.get_double("Fluid flow value");
          drained_pressure = prm.get_double("Drained pressure");
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
        double growth_incr;
        std::string  fluid_type;
        double solid_vol_frac;
        double kappa_darcy;
        double init_intrinsic_perm;
        double viscosity_FR;
        double init_darcy_coef;
        double weight_FR;
        bool gravity_term;
        int gravity_direction;
        int gravity_value;
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
                            "First Lamé parameter for extension function related to compactation point in solid material [Pa].");

          prm.declare_entry("shear modulus", "5.583e6",
                            Patterns::Double(0,1e100),
                            "shear modulus for Neo-Hooke materials [Pa].");

          prm.declare_entry("eigen solver", "QL Implicit Shifts",
                            Patterns::Selection("QL Implicit Shifts|Jacobi"),
                            "The type of eigen solver to be used for Ogden and visco-Ogden models.");

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
                            "Shear material parameter 'mu1' for first viscous mode in Ogden material [Pa].");

          prm.declare_entry("mu2_1", "0.0",
                            Patterns::Double(),
                            "Shear material parameter 'mu2' for first viscous mode in Ogden material [Pa].");

          prm.declare_entry("mu3_1", "0.0",
                            Patterns::Double(),
                            "Shear material parameter 'mu1' for first viscous mode in Ogden material [Pa].");

          prm.declare_entry("alpha1_1", "1.0",
                            Patterns::Double(),
                            "Stiffness material parameter 'alpha1' for first viscous mode in Ogden material [-].");

          prm.declare_entry("alpha2_1", "1.0",
                            Patterns::Double(),
                            "Stiffness material parameter 'alpha2' for first viscous mode in Ogden material [-].");

          prm.declare_entry("alpha3_1", "1.0",
                            Patterns::Double(),
                            "Stiffness material parameter 'alpha3' for first viscous mode in Ogden material [-].");

          prm.declare_entry("viscosity_1", "1e-10",
                            Patterns::Double(1e-10,1e100),
                            "Deformation-independent viscosity parameter 'eta_1' for first viscous mode in Ogden material [-].");

          prm.declare_entry("growth", "none",
                             Patterns::Selection("none|morphogen|pressure"),
                             "Type of continuum growth");

          prm.declare_entry("growth_incr", "1.0",
                            Patterns::Double(0,100),
                            "Morphogenetic growth increment per timestep");

          prm.declare_entry("seepage definition", "Ehlers",
                            Patterns::Selection("Markert|Ehlers"),
                            "Type of formulation used to define the seepage velocity in the problem. "
                            "Choose between Markert formulation of deformation-dependent intrinsic permeability "
                            "and Ehlers formulation of deformation-dependent Darcy flow coefficient.");

          prm.declare_entry("initial solid volume fraction", "0.67",
                            Patterns::Double(0.001,0.999),
                            "Initial porosity (solid volume fraction, 0 < n_0s < 1)");

          prm.declare_entry("kappa", "0.0",
                            Patterns::Double(0,100),
                            "Deformation-dependency control parameter for specific permeability (kappa >= 0)");

          prm.declare_entry("initial intrinsic permeability", "0.0",
                            Patterns::Double(0,1e100),
                            "Initial intrinsic permeability parameter [m^2] (isotropic permeability). To be used with Markert formulation.");

          prm.declare_entry("fluid viscosity", "0.0",
                            Patterns::Double(0, 1e100),
                            "Effective shear viscosity parameter of the fluid [Pa·s, (N·s)/m^2]. To be used with Markert formulation.");

          prm.declare_entry("initial Darcy coefficient", "1.0e-4",
                            Patterns::Double(0,1e100),
                            "Initial Darcy flow coefficient [m/s] (isotropic permeability). To be used with Ehlers formulation.");

          prm.declare_entry("fluid weight", "1.0e4",
                            Patterns::Double(0, 1e100),
                            "Effective weight of the fluid [N/m^3]. To be used with Ehlers formulation.");

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
          growth_incr =  prm.get_double("growth_incr");
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

          if ( (fluid_type == "Markert") && ((init_intrinsic_perm == 0.0) || (viscosity_FR == 0.0)) )
              throw std::runtime_error ("Markert seepage velocity formulation requires the definition of "
                                          "'initial intrinsic permeability' and 'fluid viscosity' greater than 0.0.");

          if ( (fluid_type == "Ehlers") && ((init_darcy_coef == 0.0) || (weight_FR == 0.0)) )
              throw std::runtime_error ("Ehler seepage velocity formulation requires the definition of "
                                          "'initial Darcy coefficient' and 'fluid weight' greater than 0.0.");

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
                            "Time step size. The value must be larger than the displacement error tolerance defined.");
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
        unsigned int timestep_output;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
      };

      void OutputParam::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Output parameters");
        {
          prm.declare_entry("Time step number output", "1",
                            Patterns::Integer(0),
                            "Output data for time steps multiple of the given integer value.");
        }
        prm.leave_subsection();
      }

      void OutputParam::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Output parameters");
        {
          timestep_output = prm.get_integer("Time step number output");
        }
        prm.leave_subsection();
      }

// @sect4{All parameters}
// We finally consolidate all of the above structures into a single container that holds all the run-time selections.
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
          Material_Hyperelastic( const double solid_vol_frac,
                                 const double lambda,
                                 const std::string growth_type,
                                 const double growth_incr,
                                 const Time  &time,
                                 const enum SymmetricTensorEigenvectorMethod eigen_solver)
            :
            n_OS (solid_vol_frac),
            lambda (lambda),
            growth_type(growth_type),
            growth_incr(growth_incr),
            time(time),
            growth_stretch(1.0),
            growth_stretch_converged(1.0),
            det_Fve (1.0),
            det_Fve_converged (1.0),
            eigen_solver (eigen_solver)
           {}
          ~Material_Hyperelastic()
          {}

          // Determine "extra" Kirchhoff stress as sum of isochoric and volumetric Kirchhoff stresses
          SymmetricTensor<2, dim, NumberType> get_tau_E(const Tensor<2,dim, NumberType> &F) const
          {
            //Compute (visco-elastic) part of the def. gradient tensor.
            const Tensor<2, dim> Fg = get_non_converged_growth_tensor();
            const Tensor<2, dim> Fg_inv = invert(Fg);
            const Tensor<2, dim, NumberType> Fve = F * Fg_inv;

            return ( get_tau_E_base(Fve) + get_tau_E_ext_func(Fve) ); //should it be get_tau_E_ext_func(F)?!?!
          }

          // Determine "extra" Cauchy stress as Kirchhoff stresses
          SymmetricTensor<2, dim, NumberType> get_Cauchy_E(const Tensor<2, dim, NumberType> &F) const
          {
              const NumberType det_F = determinant(F);
              Assert(det_F > 0, ExcInternalError());

              //Compute Fve
              const Tensor<2, dim> Fg = get_non_converged_growth_tensor();
              const Tensor<2, dim> Fg_inv = invert(Fg);
              const Tensor<2, dim, NumberType> Fve = F * Fg_inv;

              return get_tau_E(Fve)*NumberType(1/det_F); //Here the "whole" F is needed
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

          Tensor<2, dim> get_non_converged_growth_tensor() const
          {
              //Isotropic growth tensor
              Tensor<2, dim> Fg(Physics::Elasticity::StandardTensors<dim>::I);
              double theta = this->get_non_converged_growth_stretch();
              Fg *= theta;
              return Fg;
          }

          virtual void update_end_timestep()
          {
              det_Fve_converged = det_Fve;
              growth_stretch_converged = growth_stretch;
          }

          virtual void update_internal_equilibrium( const Tensor<2, dim, NumberType> &F )
          {
              const double det_F = Tensor<0,dim,double>(determinant(F));
              Tensor<2, dim> Fg = get_non_converged_growth_tensor();
              double det_Fg = determinant(Fg);
              det_Fve = det_F/det_Fg;

              //Growth
              this->update_growth_stretch();
          }

          virtual double get_viscous_dissipation( ) const = 0;

          // Define constitutive model parameters
          const double n_OS; //Initial porosity (solid volume fraction)
          const double lambda; //1st Lamé parameter (for extension function related to compactation point)
          const std::string growth_type;
          const double growth_incr; //Morphogenetic growth increment per timestep
          const Time  &time;

          //Internal variables
          double growth_stretch;           // Value of internal variable at this Newton step and timestep
          double growth_stretch_converged; // Value of internal variable at the previous timestep
          double det_Fve;           //Value in current iteration in current time step
          double det_Fve_converged; //Value from previous time step

          const enum SymmetricTensorEigenvectorMethod eigen_solver;

        protected:
          //Compute growth criterion
          double get_growth_criterion() const
          {
              double growth_criterion;

              if (growth_type == "none")
                  growth_criterion=0.0;

              else if (growth_type == "morphogen") //Morphogenetic growth: growth incr is const in every time step
                  growth_criterion=growth_incr;

              else
                  throw std::runtime_error ("Growth type not implemented yet.");

              return growth_criterion;

          }

          //Compute limiting function for growth
          double get_growth_limiting_function(const double &growth_stretch) const  //Not limiting in morphogeneic growth
          {
              //Silence compiler warnings
              (void)growth_stretch;
              return (1.0);
          }

          //Compute derivative of growth stretch rate (=growth_limiting_function*growth_criterion) w.r.t. growth stretch
          double get_derivative_growth_stretch_rate(const double &growth_stretch) const  //Not limiting in morphogeneic growth
          {
              //Silence compiler warnings
              (void)growth_stretch;
              return (0.0);
          }

          //Compute growth stretch
          void update_growth_stretch()
          {
              double growth_criterion = this->get_growth_criterion();
             growth_stretch = growth_stretch_converged;


              if (growth_criterion != 0.0) //If there is growth, compute growth stretch
              {
                /*
                  double growth_stretch_old = growth_stretch_converged;
                  double growth_stretch_new = growth_stretch_converged;
                  double dt = time.get_delta_t();

                  double tolerance = 1.0e-6;
                  double residual = tolerance*10.0;


                  while(abs(residual) > tolerance)
                  {
                      double growth_limiting_function = this->get_growth_limiting_function(growth_stretch_new);
                      double d_growth_stretch_rate_d_growth_stretch = this->get_derivative_growth_stretch_rate(growth_stretch_new);

                      residual = growth_stretch_old -growth_stretch_new + growth_limiting_function*growth_criterion*dt;
                      double K = 1.0 - dt*d_growth_stretch_rate_d_growth_stretch;

                      growth_stretch_old = growth_stretch_new;
                      growth_stretch_new = growth_stretch_old + residual/K;
                  }
                  growth_stretch = growth_stretch_new;
                */

                  //For morphogenic growth it's easier and faster to just write:
                  growth_stretch = growth_stretch_converged + growth_criterion;
              }
          }

          double get_non_converged_growth_stretch() const
          {
              return growth_stretch;
          }

          double get_non_converged_dgrowth_stretch_dt() const
          {
              return growth_stretch - growth_stretch_converged; //For morphogenetic growth,
          }

          // Extension function for "extra" Kirchhoff stress
          // Ehlers & Eipper 1999, doi:10.1023/A:1006565509095 --  eqn. (33)
          SymmetricTensor<2, dim, NumberType> get_tau_E_ext_func(const Tensor<2,dim, NumberType> &F) const
          {
              const NumberType det_F = determinant(F);
              Assert(det_F > 0, ExcInternalError());

              static const SymmetricTensor< 2, dim, double> I (Physics::Elasticity::StandardTensors<dim>::I);
              return  NumberType(lambda * (1.0-n_OS)*(1.0-n_OS) * (det_F/(1.0-n_OS) - det_F/(det_F-n_OS))) * I;
          }

          // Hyperelastic part of "extra" Kirchhoff stress (will be defined in each derived class)
          // Must use compressible formulation
          virtual SymmetricTensor<2, dim, NumberType> get_tau_E_base(const Tensor<2,dim, NumberType> &F) const = 0;
    };

//@sect4{Derived class: Neo-Hookean hyperelastic material}
    template <int dim, typename NumberType = Sacado::Fad::DFad<double> >
    class NeoHooke : public Material_Hyperelastic < dim, NumberType >
    {
        public:
            NeoHooke( const double solid_vol_frac,
                      const double lambda,
                      const std::string growth_type,
                      const double growth_incr,
                      const Time  &time,
                      const enum SymmetricTensorEigenvectorMethod eigen_solver,
                      const double mu )
            :
            Material_Hyperelastic< dim, NumberType > (solid_vol_frac,
                                                      lambda,
                                                      growth_type,
                                                      growth_incr,
                                                      time,
                                                      eigen_solver),
            mu(mu)
           {}
          ~NeoHooke()
          {}

           double get_viscous_dissipation() const
           {
               return 0.0;
           }

        protected:
          const double mu;

          // Hyperelastic part of "extra" Kirchhoff stress (compressible formulation)
          // Ehlers & Eipper 1999, doi:10.1023/A:1006565509095 -- eqn. (33)
          SymmetricTensor<2, dim, NumberType> get_tau_E_base(const Tensor<2,dim, NumberType> &Fve) const
          {
             static const SymmetricTensor< 2, dim, double> I (Physics::Elasticity::StandardTensors<dim>::I);

             const bool use_standard_model = true;

             if (use_standard_model)
             {
               // Standard Neo-Hooke
               return ( mu * ( symmetrize(Fve * transpose(Fve)) - I ) );
             }
             else
             {
               // Neo-Hooke in terms of principal stretches
               const SymmetricTensor<2, dim, NumberType> Bve = symmetrize(Fve * transpose(Fve));
               const std::array< std::pair< NumberType, Tensor< 1, dim, NumberType > >, dim > eigen_Bve
                 = eigenvectors(Bve, this->eigen_solver);

               SymmetricTensor<2, dim, NumberType> Bve_ev;
               for (unsigned int d=0; d<dim; ++d)
                 Bve_ev += eigen_Bve[d].first*symmetrize(outer_product(eigen_Bve[d].second,eigen_Bve[d].second));

                return mu * ( Bve_ev - I );
             }
          }
    };

//@sect4{Derived class: Ogden hyperelastic material}
    template <int dim, typename NumberType = Sacado::Fad::DFad<double> >
    class Ogden : public Material_Hyperelastic < dim, NumberType >
    {
        public:
            Ogden( const double solid_vol_frac,
                   const double lambda,
                   const std::string growth_type,
                   const double growth_incr,
                   const Time  &time,
                   const enum SymmetricTensorEigenvectorMethod eigen_solver,
                   const double mu1,
                   const double mu2,
                   const double mu3,
                   const double alpha1,
                   const double alpha2,
                   const double alpha3  )//Constructor
            :
            Material_Hyperelastic< dim, NumberType > (solid_vol_frac,
                                                      lambda,
                                                      growth_type,
                                                      growth_incr,
                                                      time,
                                                      eigen_solver),
            mu({mu1,mu2,mu3}),
            alpha({alpha1,alpha2,alpha3})
           {}
          ~Ogden()
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
          SymmetricTensor<2, dim, NumberType> get_tau_E_base(const Tensor<2,dim, NumberType> &Fve) const
          {
           //left Cauchy-Green deformation tensor
            const SymmetricTensor<2, dim, NumberType> Bve = symmetrize(Fve * transpose(Fve));

            //Compute Eigenvalues and Eigenvectors
            const std::array< std::pair< NumberType, Tensor< 1, dim, NumberType > >, dim > eigen_Bve
              = eigenvectors(Bve, this->eigen_solver);

            SymmetricTensor<2, dim, NumberType>  tau;
            static const SymmetricTensor< 2, dim, double> I (Physics::Elasticity::StandardTensors<dim>::I);

            for (unsigned int i = 0; i < 3; ++i)
            {
                for (unsigned int A = 0; A < dim; ++A)
                {
                    SymmetricTensor<2, dim, NumberType>  tau_aux1 = symmetrize(outer_product(eigen_Bve[A].second,eigen_Bve[A].second));
                    tau_aux1 *= mu[i]*std::pow(eigen_Bve[A].first, (alpha[i]/2.) );
                    tau += tau_aux1;
                }
                SymmetricTensor<2, dim, NumberType>  tau_aux2 (I);
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
    template <int dim, typename NumberType = Sacado::Fad::DFad<double> >
    class visco_Ogden : public Material_Hyperelastic < dim, NumberType >
    {
        public:
            visco_Ogden( const double solid_vol_frac,
                         const double lambda,
                         const std::string growth_type,
                         const double growth_incr,
                         const Time  &time,
                         const enum SymmetricTensorEigenvectorMethod eigen_solver,
                         const double mu1_infty,
                         const double mu2_infty,
                         const double mu3_infty,
                         const double alpha1_infty,
                         const double alpha2_infty,
                         const double alpha3_infty,
                         const double mu1_mode_1,
                         const double mu2_mode_1,
                         const double mu3_mode_1,
                         const double alpha1_mode_1,
                         const double alpha2_mode_1,
                         const double alpha3_mode_1,
                         const double viscosity_mode_1)
            :
            Material_Hyperelastic< dim, NumberType > (solid_vol_frac,
                                                      lambda,
                                                      growth_type,
                                                      growth_incr,
                                                      time,
                                                      eigen_solver),
            mu_infty({mu1_infty,mu2_infty,mu3_infty}),
            alpha_infty({alpha1_infty,alpha2_infty,alpha3_infty}),
            mu_mode_1({mu1_mode_1,mu2_mode_1,mu3_mode_1}),
            alpha_mode_1({alpha1_mode_1,alpha2_mode_1,alpha3_mode_1}),
            viscosity_mode_1(viscosity_mode_1),
            Cinv_v_1(Physics::Elasticity::StandardTensors<dim>::I),
            Cinv_v_1_converged(Physics::Elasticity::StandardTensors<dim>::I)
           {}
          ~visco_Ogden()
          {}

          void update_internal_equilibrium( const Tensor<2, dim, NumberType> &F )
          {
              Material_Hyperelastic < dim, NumberType >::update_internal_equilibrium(F);

              // Finite viscoelasticity following Reese & Govindjee (1998)
              // Algorithm for implicit exponential time integration
              // as described in Budday et al. (2017) doi: 10.1016/j.actbio.2017.06.024

              // Initialize viscous part of right Cauchy-Green deformation tensor
              this->Cinv_v_1 = this->Cinv_v_1_converged;

              //Just one Maxwell element, no for-loop needed
              //Elastic predictor step (trial values)

              //Compute Fve
              const Tensor<2, dim> Fg = this->get_non_converged_growth_tensor();
              const Tensor<2, dim> Fg_inv = invert(Fg);
              const Tensor<2, dim, NumberType> Fve = F * Fg_inv;

              //Trial elastic part of left Cauchy-Green deformation tensor
              SymmetricTensor<2, dim, NumberType> B_e_1_tr = symmetrize(Fve * this->Cinv_v_1 * transpose(Fve));

              //Compute Eigenvalues and Eigenvectors
              const std::array< std::pair< NumberType, Tensor< 1, dim, NumberType > >, dim >
                eigen_B_e_1_tr = eigenvectors(B_e_1_tr, this->eigen_solver);

              Tensor< 1, dim, NumberType > lambdas_e_1_tr;
              Tensor< 1, dim, NumberType > epsilon_e_1_tr;
              for (int a = 0; a < dim; ++a)
              {
                  //Trial elastic principal stretches
                  lambdas_e_1_tr[a] = std::sqrt(eigen_B_e_1_tr[a].first);

                  //Trial elastic logarithmic principal stretches
                  epsilon_e_1_tr[a] = std::log(lambdas_e_1_tr[a]);
              }

             //Inelastic corrector step
             const double tolerance = 1e-8;
             double residual_check = tolerance*10.0;
             Tensor< 1, dim, NumberType > residual;
             Tensor< 2, dim, NumberType > tangent;
             static const SymmetricTensor< 2, dim, double> I(Physics::Elasticity::StandardTensors<dim>::I);
             NumberType J_e_1 = std::sqrt(determinant(B_e_1_tr));

             std::vector<NumberType> lambdas_e_1_iso(dim);
             SymmetricTensor<2, dim, NumberType> B_e_1;
             int iteration = 0;

             Tensor< 1, dim, NumberType > lambdas_e_1;
             Tensor< 1, dim, NumberType > epsilon_e_1;
             epsilon_e_1 = epsilon_e_1_tr;

              while(residual_check > tolerance)
              {
                  NumberType aux_J_e_1 = 1.0;
                  for (unsigned int a = 0; a < dim; ++a)
                  {
                      lambdas_e_1[a] = std::exp(epsilon_e_1[a]);
                      aux_J_e_1 *= lambdas_e_1[a];
                  }

                  J_e_1 = aux_J_e_1;

                  for (unsigned int a = 0; a < dim; ++a)
                      lambdas_e_1_iso[a] = lambdas_e_1[a]*std::pow(J_e_1,-1.0/dim);

                  for (unsigned int a = 0; a < dim; ++a)
                  {
                      residual[a] = get_beta_mode_1(lambdas_e_1_iso, a);
                      residual[a] *= this->time.get_delta_t()/(2.0*viscosity_mode_1);
                      residual[a] += epsilon_e_1[a];
                      residual[a] -= epsilon_e_1_tr[a];

                      for (unsigned int b = 0; b < dim; ++b)
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
                  for (unsigned int a = 0; a < dim; ++a)
                  {
                      if ( std::abs(residual[a]) > residual_check)
                          residual_check = std::abs(Tensor<0,dim,double>(residual[a]));
                  }
                  iteration += 1;
                  if (iteration > 15 )
                      throw std::runtime_error ("No convergence in local Newton iteration for the "
                                                "viscoelastic exponential time integration algorithm.");
              }

              //Compute converged stretches and left Cauchy-Green deformation tensor of mode 1
              NumberType aux_J_e_1 = 1.0;
              for (unsigned int a = 0; a < dim; ++a)
              {
                  lambdas_e_1[a] = std::exp(epsilon_e_1[a]);
                  aux_J_e_1 *= lambdas_e_1[a];
              }
              J_e_1 = aux_J_e_1;

              for (unsigned int a = 0; a < dim; ++a)
                  lambdas_e_1_iso[a] = lambdas_e_1[a]*std::pow(J_e_1,-1.0/dim);

              for (unsigned int a = 0; a < dim; ++a)
              {
                  SymmetricTensor<2, dim, NumberType>
                  B_e_1_aux = symmetrize(outer_product(eigen_B_e_1_tr[a].second,eigen_B_e_1_tr[a].second));
                  B_e_1_aux *= lambdas_e_1[a] * lambdas_e_1[a];
                  B_e_1 += B_e_1_aux;
              }

              //Update inverse of the viscous right Cauchy-Green deformation tensor of mode 1
              Tensor<2, dim, NumberType>Cinv_v_1_AD = symmetrize(invert(F) * B_e_1 * invert(transpose(F)));

              //Update tau_E_neq_1
              this->tau_neq_1 = 0;
              for (unsigned int a = 0; a < dim; ++a)
              {
                  SymmetricTensor<2, dim, NumberType>
                  tau_neq_1_aux = symmetrize(outer_product(eigen_B_e_1_tr[a].second,eigen_B_e_1_tr[a].second));
                  tau_neq_1_aux *=  get_beta_mode_1(lambdas_e_1_iso, a);
                  this->tau_neq_1 += tau_neq_1_aux;
              }

              // Store history
              for (unsigned int a = 0; a < dim; ++a)
                  for (unsigned int b = 0; b < dim; ++b)
                      this->Cinv_v_1[a][b]= Tensor<0,dim,double>(Cinv_v_1_AD[a][b]);
          }

          void update_end_timestep()
          {
              Material_Hyperelastic < dim, NumberType >::update_end_timestep();
              this->Cinv_v_1_converged = this->Cinv_v_1;
          }

           double get_viscous_dissipation() const
           {
               NumberType dissipation_term = get_tau_E_neq() * get_tau_E_neq(); //Double contract the two SymmetricTensor
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
          SymmetricTensor<2, dim, NumberType> get_tau_E_base(const Tensor<2,dim, NumberType> &Fve) const
          {
              return ( get_tau_E_neq() + get_tau_E_eq(Fve) );
          }

          // Equilibrium (hyperelastic) part of "extra" Kirchhoff stress
          SymmetricTensor<2, dim, NumberType> get_tau_E_eq(const Tensor<2,dim, NumberType> &Fve) const
          {
            //left Cauchy-Green deformation tensor
            const SymmetricTensor<2, dim, NumberType> Bve = symmetrize(Fve * transpose(Fve));

            //Compute Eigenvalues and Eigenvectors
            std::array< std::pair< NumberType, Tensor< 1, dim, NumberType > >, dim > eigen_Bve;
            eigen_Bve = eigenvectors(Bve, this->eigen_solver);

            SymmetricTensor<2, dim, NumberType>  tau;
            static const SymmetricTensor< 2, dim, double> I (Physics::Elasticity::StandardTensors<dim>::I);

            for (unsigned int i = 0; i < 3; ++i)
            {
                for (unsigned int A = 0; A < dim; ++A)
                {
                    SymmetricTensor<2, dim, NumberType>  tau_aux1 = symmetrize(outer_product(eigen_Bve[A].second,eigen_Bve[A].second));
                    tau_aux1 *= mu_infty[i]*std::pow(eigen_Bve[A].first, (alpha_infty[i]/2.) );
                    tau += tau_aux1;
                }
                SymmetricTensor<2, dim, NumberType>  tau_aux2 (I);
                tau_aux2 *= mu_infty[i];
                tau -= tau_aux2;
            }
            return tau;
          }

          SymmetricTensor<2, dim, NumberType> get_tau_E_neq() const
          {
              return tau_neq_1;
          }

          //Compute beta term for the given (volume invariant) stretches
          NumberType get_beta_mode_1(std::vector< NumberType > &lambda_ve, const int &A) const
          {
              NumberType beta = 0.0;

              for (unsigned int i = 0; i < 3; ++i) //3rd-order Ogden model
              {

                  NumberType aux = 0.0;
                  for (int p = 0; p < dim; ++p)
                      aux += std::pow(lambda_ve[p],alpha_mode_1[i]);

                  aux *= -1.0/dim;
                  aux += std::pow(lambda_ve[A], alpha_mode_1[i]);
                  aux *= mu_mode_1[i];

                  beta  += aux;
              }
              return beta;
          }

          //Compute gamma term for the given (volume invariant) stretches
          NumberType get_gamma_mode_1(std::vector< NumberType > &lambda_ve, const int &A, const int &B) const
          {
              NumberType gamma = 0.0;

              if (A==B)
              {
                  for (unsigned int i = 0; i < 3; ++i)
                  {
                      NumberType aux = 0.0;
                      for (int p = 0; p < dim; ++p)
                          aux += std::pow(lambda_ve[p],alpha_mode_1[i]);

                      aux *= 1.0/(dim*dim);
                      aux += 1.0/dim * std::pow(lambda_ve[A], alpha_mode_1[i]);
                      aux *= mu_mode_1[i]*alpha_mode_1[i];

                      gamma += aux;
                  }
              }
              else
              {
                  for (unsigned int i = 0; i < 3; ++i)
                  {
                      NumberType aux = 0.0;
                      for (int p = 0; p < dim; ++p)
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
    template <int dim, typename NumberType = Sacado::Fad::DFad<double> >
    class Material_Darcy_Fluid
    {
       public:
         Material_Darcy_Fluid(const std::string  fluid_type,
                              const double solid_vol_frac,
                              const double init_intrinsic_perm,
                              const double viscosity_FR,
                              const double init_darcy_coef,
                              const double weight_FR,
                              const double kappa_darcy,
                              const bool gravity_term,
                              const double density_FR,
                              const int gravity_direction,
                              const double gravity_value)
           :
           fluid_type(fluid_type),
           n_OS(solid_vol_frac),
           initial_intrinsic_permeability(init_intrinsic_perm),
           viscosity_FR(viscosity_FR),
           initial_darcy_coefficient(init_darcy_coef),
           weight_FR(weight_FR),
           kappa_darcy(kappa_darcy),
           gravity_term(gravity_term),
           density_FR(density_FR),
           gravity_direction(gravity_direction),
           gravity_value(gravity_value)
         {
           Assert(kappa_darcy >= 0, ExcInternalError());
         }

         ~Material_Darcy_Fluid()
         {}

         Tensor<1, dim, NumberType> get_seepage_velocity_current (const Tensor<2,dim, NumberType> &F,
                                                                  const Tensor<1,dim, NumberType> &grad_p_fluid) const
         {
             const NumberType det_F = determinant(F);
             Assert(det_F > 0.0, ExcInternalError());

             Tensor<2, dim, NumberType> permeability_term;

             if (fluid_type == "Markert")
                 permeability_term = get_instrinsic_permeability_current(F) / viscosity_FR;

             else if (fluid_type == "Ehlers")
                 permeability_term = get_darcy_flow_current(F) / weight_FR;

             else
                 throw std::runtime_error ("Material_Darcy_Fluid --> Only Markert and Ehlers formulations have been implemented.");

             return -1.0 * permeability_term * det_F * (grad_p_fluid - get_body_force_FR_current());
         }

         double get_porous_dissipation(const Tensor<2,dim, NumberType> &F,
                                       const Tensor<1,dim, NumberType> &grad_p_fluid) const
         {
             NumberType dissipation_term;
             Tensor<1, dim, NumberType> seepage_velocity;
             Tensor<2, dim, NumberType> permeability_term;

             const NumberType det_F = determinant(F);
             Assert(det_F > 0.0, ExcInternalError());

             if (fluid_type == "Markert")
             {
                 permeability_term = get_instrinsic_permeability_current(F) / viscosity_FR;
                 seepage_velocity = get_seepage_velocity_current(F,grad_p_fluid);
             }
             else if (fluid_type == "Ehlers")
             {
                 permeability_term = get_darcy_flow_current(F) / weight_FR;
                 seepage_velocity = get_seepage_velocity_current(F,grad_p_fluid);
             }
             else
                 throw std::runtime_error ("Material_Darcy_Fluid --> Only Markert and Ehlers formulations have been implemented.");

             dissipation_term = ( invert(permeability_term) * seepage_velocity ) * seepage_velocity;
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
         const int    gravity_value;

         Tensor<2, dim, NumberType> get_instrinsic_permeability_current(const Tensor<2,dim, NumberType> &F) const
         {
           static const SymmetricTensor< 2, dim, double> I (Physics::Elasticity::StandardTensors<dim>::I);
           const Tensor<2, dim, NumberType> initial_instrinsic_permeability_tensor = Tensor<2, dim, double>(initial_intrinsic_permeability * I);

           const NumberType det_F = determinant(F);
           Assert(det_F > 0.0, ExcInternalError());

           const NumberType fraction = (det_F - n_OS)/(1 - n_OS);
           return NumberType (std::pow(fraction, kappa_darcy)) * initial_instrinsic_permeability_tensor;
         }

         Tensor<2, dim, NumberType> get_darcy_flow_current(const Tensor<2,dim, NumberType> &F) const
         {
           static const SymmetricTensor< 2, dim, double> I (Physics::Elasticity::StandardTensors<dim>::I);
           const Tensor<2, dim, NumberType> initial_darcy_flow_tensor = Tensor<2, dim, double>(initial_darcy_coefficient * I);

           const NumberType det_F = determinant(F);
           Assert(det_F > 0.0, ExcInternalError());

           const NumberType fraction = (1.0 - (n_OS / det_F) )/(1.0 - n_OS);
           return NumberType (std::pow(fraction, kappa_darcy)) * initial_darcy_flow_tensor;
         }

        Tensor<1, dim, NumberType> get_body_force_FR_current() const
        {
            Tensor<1, dim, NumberType> body_force_FR_current;

            if (gravity_term == true)
            {
               Tensor<1, dim, NumberType> gravity_vector;
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
    template <int dim, typename NumberType = Sacado::Fad::DFad<double> > //double>
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
                    solid_material.reset(new NeoHooke <dim, NumberType>(parameters.solid_vol_frac,
                                                                        parameters.lambda,
                                                                        parameters.growth_type,
                                                                        parameters.growth_incr,
                                                                        time,
                                                                        parameters.eigen_solver,
                                                                        parameters.mu));
                else if (parameters.mat_type == "Ogden")
                    solid_material.reset(new Ogden <dim, NumberType>(parameters.solid_vol_frac,
                                                                     parameters.lambda,
                                                                     parameters.growth_type,
                                                                     parameters.growth_incr,
                                                                     time,
                                                                     parameters.eigen_solver,
                                                                     parameters.mu1_infty,
                                                                     parameters.mu2_infty,
                                                                     parameters.mu3_infty,
                                                                     parameters.alpha1_infty,
                                                                     parameters.alpha2_infty,
                                                                     parameters.alpha3_infty));
                else if (parameters.mat_type == "visco-Ogden")
                    solid_material.reset(new visco_Ogden <dim, NumberType>(parameters.solid_vol_frac,
                                                                           parameters.lambda,
                                                                           parameters.growth_type,
                                                                           parameters.growth_incr,
                                                                           time,
                                                                           parameters.eigen_solver,
                                                                           parameters.mu1_infty,
                                                                           parameters.mu2_infty,
                                                                           parameters.mu3_infty,
                                                                           parameters.alpha1_infty,
                                                                           parameters.alpha2_infty,
                                                                           parameters.alpha3_infty,
                                                                           parameters.mu1_mode_1,
                                                                           parameters.mu2_mode_1,
                                                                           parameters.mu3_mode_1,
                                                                           parameters.alpha1_mode_1,
                                                                           parameters.alpha2_mode_1,
                                                                           parameters.alpha3_mode_1,
                                                                           parameters.viscosity_mode_1));
                else
                    Assert (false, ExcMessage("Material type not implemented"));

                fluid_material.reset(new Material_Darcy_Fluid<dim, NumberType> (parameters.fluid_type,
                                                                                parameters.solid_vol_frac,
                                                                                parameters.init_intrinsic_perm,
                                                                                parameters.viscosity_FR,
                                                                                parameters.init_darcy_coef,
                                                                                parameters.weight_FR,
                                                                                parameters.kappa_darcy,
                                                                                parameters.gravity_term,
                                                                                parameters.density_FR,
                                                                                parameters.gravity_direction,
                                                                                parameters.gravity_value));
            }

            // We offer an interface to retrieve certain data (used in the material and
            // global tangent matrix and residual assembly operations)
            SymmetricTensor<2, dim, NumberType> get_tau_E(const Tensor<2, dim, NumberType> &Fve) const
            {
                return solid_material->get_tau_E(Fve);
            }

            SymmetricTensor<2, dim, NumberType>  get_Cauchy_E(const Tensor<2, dim, NumberType> &Fve) const
            {
                return solid_material->get_Cauchy_E(Fve);
            }

            double get_converged_det_Fve() const
            {
              return  solid_material->get_converged_det_Fve();
            }

            double get_converged_growth_stretch() const
            {
              return  solid_material->get_converged_growth_stretch();
            }

            Tensor<2, dim> get_non_converged_growth_tensor() const
            {
              return  solid_material->get_non_converged_growth_tensor();
            }

            void update_end_timestep()
            {
                solid_material->update_end_timestep();
            }

            void update_internal_equilibrium(const Tensor<2, dim, NumberType> &F )
            {
                solid_material->update_internal_equilibrium(F);
            }

            double get_viscous_dissipation() const
            {
                return solid_material->get_viscous_dissipation();
            }

            Tensor<1,dim, NumberType> get_seepage_velocity_current (const Tensor<2,dim, NumberType> &F,
                                                                    const Tensor<1,dim, NumberType> &grad_p_fluid) const
             {
                 return fluid_material->get_seepage_velocity_current(F, grad_p_fluid);
             }

            double get_porous_dissipation(const Tensor<2,dim, NumberType> &F,
                                           const Tensor<1,dim, NumberType> &grad_p_fluid) const
            {
                return fluid_material->get_porous_dissipation(F, grad_p_fluid);
            }

            Tensor<1, dim, NumberType> get_overall_body_force (const Tensor<2,dim, NumberType> &F,
                                                               const Parameters::AllParameters &parameters) const
            {
                Tensor<1, dim, NumberType> body_force;

                if (parameters.gravity_term == true)
                {
                    const NumberType det_F_AD = determinant(F);
                    Assert(det_F_AD > 0.0, ExcInternalError());

                    const NumberType overall_density_ref = parameters.density_SR * parameters.solid_vol_frac
                                                  + parameters.density_FR * (det_F_AD - parameters.solid_vol_frac);

                   Tensor<1, dim, NumberType> gravity_vector;
                   gravity_vector[parameters.gravity_direction] = parameters.gravity_value;
                   body_force = overall_density_ref * gravity_vector;
                }

                return body_force;
            }
        private:
            std_cxx11::shared_ptr< Material_Hyperelastic<dim, NumberType> > solid_material;
            std_cxx11::shared_ptr< Material_Darcy_Fluid<dim, NumberType> > fluid_material;
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
            typedef Sacado::Fad::DFad<double> ADNumberType;

            std::ofstream outfile;
            std::ofstream pointfile;

            struct PerTaskData_ASM;
            template<typename NumberType = double> struct ScratchData_ASM;

            //Generate mesh
            virtual void make_grid() = 0;

            //Define points for post-processing
            virtual void define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices) = 0;

            //Set up the finite element system to be solved:
            void system_setup(TrilinosWrappers::MPI::BlockVector &solution_delta_OUT);

            //Extract sub-blocks from the global matrix
            void determine_component_extractors();

            // Several functions to assemble the system and right hand side matrices using multithreading.
            void assemble_system( const TrilinosWrappers::MPI::BlockVector &solution_delta_OUT );
            void assemble_system_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                          ScratchData_ASM<ADNumberType> &scratch,
                                          PerTaskData_ASM &data) const;
            void copy_local_to_global_system(const PerTaskData_ASM &data);

            // Define boundary conditions
            virtual void make_constraints(const int &it_nr);
            virtual void make_dirichlet_constraints(ConstraintMatrix &constraints) = 0;
            virtual Tensor<1,dim> get_neumann_traction (const types::boundary_id &boundary_id,
                                                        const Point<dim>         &pt,
                                                        const Tensor<1,dim>      &N) const = 0;
            virtual double get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                                           const Point<dim>         &pt) const = 0;
            virtual types::boundary_id get_reaction_boundary_id_for_output () const = 0;
            virtual  std::pair<types::boundary_id,types::boundary_id> get_drained_boundary_id_for_output () const = 0;
            virtual std::pair<double, FEValuesExtractors::Scalar> get_dirichlet_load(const types::boundary_id &boundary_id) const = 0;

            // Create and update the quadrature points.
            void setup_qph();

            //Solve non-linear system using a Newton-Raphson scheme
            void solve_nonlinear_timestep(TrilinosWrappers::MPI::BlockVector &solution_delta_OUT);

            //Solve the linearized equations using a direct solver
            void solve_linear_system ( TrilinosWrappers::MPI::BlockVector &newton_update_OUT);

            //Retrieve the  solution
            TrilinosWrappers::MPI::BlockVector get_total_solution(const TrilinosWrappers::MPI::BlockVector &solution_delta_IN) const;

            // Store the converged values of the internal variables at the end of each timestep
            void update_end_timestep();

            //Post-processing and writing data to files
            void output_results(const unsigned int timestep,
                                const double current_time,
                                TrilinosWrappers::MPI::BlockVector solution,
                                std::vector<Point<dim> > &tracked_vertices,
                                std::ofstream &pointfile) const;

            // Headers and footer for the output files
            void print_console_file_header( std::ofstream &outfile) const;
            void print_plot_file_header(std::vector<Point<dim> > &tracked_vertices,
                                        std::ofstream &pointfile) const;
            void print_console_file_footer(std::ofstream &outfile) const;
            void print_plot_file_footer( std::ofstream &pointfile) const;

            // For parallel communication
            MPI_Comm                         mpi_communicator;
            const unsigned int               n_mpi_processes;
            const unsigned int               this_mpi_process;
            mutable ConditionalOStream       pcout;

            //A collection of the parameters used to describe the problem setup
            const Parameters::AllParameters &parameters;

            //Declare an instance of dealii Triangulation class (mesh)
            Triangulation<dim>  triangulation;

            // Keep track of the current time and the time spent evaluating certain functions
            Time          time;
            TimerOutput   timerconsole;
            TimerOutput   timerfile;

            // A storage object for quadrature point information.
            CellDataStorage<typename Triangulation<dim>::cell_iterator, PointHistory<dim,ADNumberType> > quadrature_point_history;

            //Integers to store polynomial degree (needed for output)
            const unsigned int  degree_displ;
            const unsigned int  degree_pore;

            //Declare an instance of dealii FESystem class (finite element definition)
            const FESystem<dim> fe;

            //Declare an instance of dealii DoFHandler class (assign DoFs to mesh)
            DoFHandler<dim>     dof_handler_ref;

            //Integer to store DoFs per element (this value will be used often)
            const unsigned int  dofs_per_cell;

            //Declare an instance of dealii Extractor objects used to retrieve information from the solution vectors
            //We will use "u_fe" and "p_fluid_fe"as subscript in operator [] expressions on FEValues and FEFaceValues
            //objects to extract the components of the displacement vector and fluid pressure, respectively.
            const FEValuesExtractors::Vector u_fe;
            const FEValuesExtractors::Scalar p_fluid_fe;

            // Description of how the block-system is arranged. There are 3 blocks:
            //   0 - vector DOF displacements u
            //   1 - scalar DOF fluid pressure p_fluid
            static const unsigned int  n_blocks = 2;
            static const unsigned int  n_components = dim+1;
            static const unsigned int  first_u_component = 0;
            static const unsigned int  p_fluid_component = dim;

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

            std::vector<types::global_dof_index>          dofs_per_block;
            std::vector<types::global_dof_index>        element_indices_u;
            std::vector<types::global_dof_index>        element_indices_p_fluid;

            //Declare an instance of dealii QGauss class (The Gauss-Legendre family of quadrature rules for numerical integration)
            //Gauss Points in element, with n quadrature points (in each space direction <dim> )
            const QGauss<dim>                qf_cell;
            //Gauss Points on element faces (used for definition of BCs)
            const QGauss<dim - 1>            qf_face;
            //Integer to store num GPs per element (this value will be used often)
            const unsigned int               n_q_points;
            //Integer to store num GPs per face (this value will be used often)
            const unsigned int               n_q_points_f;

            //Declare an instance of dealii ConstraintMatrix class (linear constraints on DoFs due to hanging nodes or BCs)
            ConstraintMatrix          constraints;

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
            void get_error_update(const TrilinosWrappers::MPI::BlockVector &newton_update_IN, Errors &error_update_OUT);

            // Print information to screen
            void print_conv_header();
            void print_conv_footer();

//NOTE: In all functions, we pass by reference (&), so these functions work on the original copy (not a clone copy),
//      modifying the input variables inside the functions will change them outside the function.
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
        triangulation(Triangulation<dim>::maximum_smoothing),
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
        p_fluid_fe(p_fluid_component),
        x_displacement(first_u_component),
        y_displacement(first_u_component+1),
        z_displacement(first_u_component+2),
        pressure(p_fluid_component),
        dofs_per_block(n_blocks),
        qf_cell(parameters.quad_order),
        qf_face(parameters.quad_order),
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
          std::vector<Point<dim> > tracked_vertices (2);
          define_tracked_vertices(tracked_vertices);
          std::vector<Point<dim>> reaction_force;

          if (this_mpi_process == 0)
          {
              pointfile.open("data-for-gnuplot.sol");
              print_plot_file_header(tracked_vertices, pointfile);
          }

          //Print results to output file
          output_results(time.get_timestep(), time.get_current(), solution_n, tracked_vertices, pointfile);

          //Increment time step (=load step)
          //NOTE: In solving the quasi-static problem, the time becomes a loading parameter,
          //i.e. we increase the loading linearly with time, making the two concepts interchangeable.
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

          while ( (time.get_end() - time.get_current()) > -1.0*parameters.tol_u )
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
              if ( ( time.get_timestep() % parameters.timestep_output ) == 0 )
                  output_results(time.get_timestep(), time.get_current(), solution_n, tracked_vertices, pointfile);

              //Increment the time step (=load step)
              time.increment_time();
            }

          //Print the footers and close files
          if (this_mpi_process == 0)
          {
              print_plot_file_footer(pointfile);
              pointfile.close ();
              print_console_file_footer(outfile);

              //NOTE: ideally, we should close the outfile here [ >> outfile.close (); ]
              //But if we do, then the timer output will not be printed. That is why we leave it open.
          }
    }

// @sect4{Private interface}
// We define the structures needed for parallelization with Threading Building Blocks (TBB)
// Tangent matrix and right-hand side force vector assembly structures.
// PerTaskData_ASM stores local contributions
    template <int dim>
    struct Solid<dim>::PerTaskData_ASM
    {
        FullMatrix<double>        cell_matrix;
        Vector<double>            cell_rhs;
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
        std::vector<NumberType>                  local_dof_values;
        std::vector<Tensor<2, dim, NumberType> > solution_grads_u_total;
        std::vector<NumberType>                  solution_values_p_fluid_total;
        std::vector<Tensor<1, dim, NumberType> > solution_grads_p_fluid_total;
        std::vector<Tensor<1, dim, NumberType> > solution_grads_face_p_fluid_total;

        //shape function values
        std::vector<std::vector<Tensor<1,dim>>>          Nx;
        std::vector<std::vector<double>>                 Nx_p_fluid;
        //shape function gradients
        std::vector<std::vector<Tensor<2,dim, NumberType>>>          grad_Nx;
        std::vector<std::vector<SymmetricTensor<2,dim, NumberType>>> symm_grad_Nx;
        std::vector<std::vector<Tensor<1,dim, NumberType>>>          grad_Nx_p_fluid;

        ScratchData_ASM(const FiniteElement<dim> &fe_cell,
                        const QGauss<dim> &qf_cell, const UpdateFlags uf_cell,
                        const QGauss<dim - 1> & qf_face, const UpdateFlags uf_face,
                        const TrilinosWrappers::MPI::BlockVector &solution_total    )
          :
          solution_total (solution_total),
          fe_values_ref(fe_cell, qf_cell, uf_cell),
          fe_face_values_ref(fe_cell, qf_face, uf_face),
          local_dof_values(fe_cell.dofs_per_cell),
          solution_grads_u_total(qf_cell.size()),
          solution_values_p_fluid_total(qf_cell.size()),
          solution_grads_p_fluid_total(qf_cell.size()),
          solution_grads_face_p_fluid_total(qf_face.size()),
          Nx(qf_cell.size(), std::vector<Tensor<1,dim>>(fe_cell.dofs_per_cell)),
          Nx_p_fluid(qf_cell.size(), std::vector<double>(fe_cell.dofs_per_cell)),
          grad_Nx(qf_cell.size(), std::vector<Tensor<2, dim, NumberType>>(fe_cell.dofs_per_cell)),
          symm_grad_Nx(qf_cell.size(), std::vector<SymmetricTensor<2, dim, NumberType>> (fe_cell.dofs_per_cell)),
          grad_Nx_p_fluid(qf_cell.size(), std::vector<Tensor<1, dim, NumberType>>(fe_cell.dofs_per_cell))
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
          solution_values_p_fluid_total(rhs.solution_values_p_fluid_total),
          solution_grads_p_fluid_total(rhs.solution_grads_p_fluid_total),
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

          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              local_dof_values[k] = 0.0;
            }

          Assert(solution_grads_u_total.size() == n_q_points, ExcInternalError());
          Assert(solution_values_p_fluid_total.size() == n_q_points, ExcInternalError());
          Assert(solution_grads_p_fluid_total.size() == n_q_points, ExcInternalError());

          Assert(Nx.size() == n_q_points, ExcInternalError());
          Assert(grad_Nx.size() == n_q_points, ExcInternalError());
          Assert(symm_grad_Nx.size() == n_q_points, ExcInternalError());

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              Assert( Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
              Assert( grad_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
              Assert( symm_grad_Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());

              solution_grads_u_total[q_point] = 0.0;
              solution_values_p_fluid_total[q_point] = 0.0;
              solution_grads_p_fluid_total[q_point] = 0.0;

              for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
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

          for (unsigned int f_q_point = 0; f_q_point < n_f_q_points; ++f_q_point)
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

        // Partition triangulation
        GridTools::partition_triangulation (n_mpi_processes, triangulation);


        //Determine number of components per block
        std::vector<unsigned int> block_component(n_components, u_block);
        block_component[p_fluid_component] = p_fluid_block;

        // The DOF handler is initialised and we renumber the grid in an efficient manner.
        dof_handler_ref.distribute_dofs(fe);
        DoFRenumbering::Cuthill_McKee(dof_handler_ref);
        DoFRenumbering::component_wise(dof_handler_ref, block_component);

        // Count the number of DoFs in each block
        dofs_per_block.clear();
        dofs_per_block.resize(n_blocks);
        DoFTools::count_dofs_per_block(dof_handler_ref, dofs_per_block, block_component);

        // Setup the sparsity pattern and tangent matrix
        all_locally_owned_dofs = DoFTools::locally_owned_dofs_per_subdomain (dof_handler_ref);
        std::vector<IndexSet> all_locally_relevant_dofs
        = DoFTools::locally_relevant_dofs_per_subdomain (dof_handler_ref);

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
            locally_owned_partitioning.push_back(locally_owned_dofs.get_view(idx_begin, idx_end));
            locally_relevant_partitioning.push_back(locally_relevant_dofs.get_view(idx_begin, idx_end));
          }

        //Print information on screen
        pcout  << "\nTriangulation:\n" << "  Number of active cells: " << triangulation.n_active_cells()  << " (by partition:";
        for (unsigned int p=0; p<n_mpi_processes; ++p)
          pcout  << (p==0 ? ' ' : '+')
          << (GridTools::count_cells_with_subdomain_association (triangulation,p));
        pcout << ")" << std::endl;
        pcout << "  Number of degrees of freedom: " << dof_handler_ref.n_dofs() << " (by partition:";
        for (unsigned int p=0; p<n_mpi_processes; ++p)
          pcout  << (p==0 ? ' ' : '+')
          << (DoFTools::count_dofs_with_subdomain_association (dof_handler_ref,p));
        pcout << ")" << std::endl;
        pcout   << "  Number of degrees of freedom per block: "
            << "[n_u, n_p_fluid] = ["
            << dofs_per_block[u_block] << ", "
            << dofs_per_block[p_fluid_block] << "]"
            << std::endl;

        //Print information to file
        outfile  << "\nTriangulation:\n" <<  "  Number of active cells: " << triangulation.n_active_cells()  << " (by partition:";
        for (unsigned int p=0; p<n_mpi_processes; ++p)
          outfile << (p==0 ? ' ' : '+')
          << (GridTools::count_cells_with_subdomain_association (triangulation,p));
        outfile << ")" << std::endl;
        outfile << "  Number of degrees of freedom: " << dof_handler_ref.n_dofs() << " (by partition:";
        for (unsigned int p=0; p<n_mpi_processes; ++p)
          outfile  << (p==0 ? ' ' : '+')
          << (DoFTools::count_dofs_with_subdomain_association (dof_handler_ref,p));
        outfile << ")" << std::endl;
        outfile << "  Number of degrees of freedom per block: "
            << "[n_u, n_p_fluid] = ["
            << dofs_per_block[u_block] << ", "
            << dofs_per_block[p_fluid_block] << "]"
            << std::endl;


        // We optimise the sparsity pattern to reflect this structure and prevent
        // unnecessary data creation for the right-diagonal block components.
        Table<2, DoFTools::Coupling> coupling(n_components, n_components);
        for (unsigned int ii = 0; ii < n_components; ++ii)
          for (unsigned int jj = 0; jj < n_components; ++jj)

            //Identify "zero" matrix components of FE-system (The two components do not couple)
            if (((ii == p_fluid_component) && (jj < p_fluid_component))
                || ((ii < p_fluid_component) && (jj == p_fluid_component)) )
              coupling[ii][jj] = DoFTools::none;

            //The rest of components always couple
            else
              coupling[ii][jj] = DoFTools::always;

        TrilinosWrappers::BlockSparsityPattern bsp (locally_owned_partitioning,
                                                    mpi_communicator);

        DoFTools::make_sparsity_pattern (dof_handler_ref, bsp, constraints, false, this_mpi_process);
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
        DoFTools::make_sparsity_pattern (dof_handler_ref, sp, constraints, false, this_mpi_process);
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

        for (unsigned int k = 0; k < fe.dofs_per_cell; ++k)
          {
            const unsigned int k_group = fe.system_to_base_index(k).first.first;
            if (k_group == u_block)
              element_indices_u.push_back(k);
            else if (k_group == p_fluid_block)
              element_indices_p_fluid.push_back(k);
            else
              {
                Assert(k_group <= p_fluid_block, ExcInternalError());
              }
          }
    }

    //Set-up quadrature point history (QPH) data objects
    template <int dim>
    void Solid<dim>::setup_qph()
    {
        pcout       << "\nSetting up quadrature point data..." << std::endl;
        outfile   << "\nSetting up quadrature point data..." << std::endl;

        //Create QPH data objects.
        quadrature_point_history.initialize(triangulation.begin_active(), triangulation.end(), n_q_points);

        //Setup the initial quadrature point data using the info stored in parameters
        FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
        cell (IteratorFilters::SubdomainEqualTo(this_mpi_process), dof_handler_ref.begin_active()),
        endc (IteratorFilters::SubdomainEqualTo(this_mpi_process), dof_handler_ref.end());
        for (; cell!=endc; ++cell)
          {
            Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
            const std::vector<std::shared_ptr<PointHistory<dim, ADNumberType> > >
                lqph = quadrature_point_history.get_data(cell);
            Assert(lqph.size() == n_q_points, ExcInternalError());

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              lqph[q_point]->setup_lqp(parameters, time);
          }
    }

    //Solve the non-linear system using a Newton-Raphson scheme
    template <int dim>
    void Solid<dim>::solve_nonlinear_timestep(TrilinosWrappers::MPI::BlockVector &solution_delta_OUT)
    {
        //Print the load step
        pcout       << std::endl << "\nTimestep " << time.get_timestep() << " @ "
            << time.get_current() << "s" << std::endl;
        outfile   << std::endl << "\nTimestep " << time.get_timestep() << " @ "
            << time.get_current() << "s" << std::endl;

        //Declare newton_update vector (solution of a Newton iteration),
        //which must have as many positions as global DoFs.
        TrilinosWrappers::MPI::BlockVector newton_update(locally_owned_partitioning, mpi_communicator);

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
        while(newton_iteration < parameters.max_iterations_NR)
          {
            pcout       << " " << std::setw(2) << newton_iteration << " " << std::flush;
            outfile   << " " << std::setw(2) << newton_iteration << " " << std::flush;

            //Initialize global stiffness matrix and global force vector to zero
            tangent_matrix = 0.0;
            system_rhs = 0.0;

            tangent_matrix_nb = 0.0;
            system_rhs_nb = 0.0;

            //Apply boundary conditions
            make_constraints(newton_iteration);

            assemble_system(solution_delta_OUT);

            //Compute the rhs residual (error between external and internal forces in FE system)
            get_error_residual(error_residual);

            //error_residual in first iteration is stored to normalize posterior error measures
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
                        << "  " << error_residual_norm.norm << "  " << error_residual_norm.u
                        << "  " << error_residual_norm.p_fluid
                        << "        " << error_update_norm.norm  << "  " << error_update_norm.u
                        << "  " << error_update_norm.p_fluid
                        << "  " << std::endl;
                outfile   << "\n ***** CONVERGED! *****     "
                        << system_rhs.l2_norm() << "      "
                        << "  " << error_residual_norm.norm << "  " << error_residual_norm.u
                        << "  " << error_residual_norm.p_fluid
                        << "        " << error_update_norm.norm  << "  " << error_update_norm.u
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

            //error_update in first iteration is stored to normalize posterior error measures
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
            << "        " << error_residual_norm.norm << "  " << error_residual_norm.u
            << "  " << error_residual_norm.p_fluid
            << "        " << error_update_norm.norm  << "  " << error_update_norm.u
            << "  " << error_update_norm.p_fluid
            << "  " << std::endl;

            outfile  << " |   " << std::fixed << std::setprecision(3)
            << std::setw(7) << std::scientific
            << system_rhs.l2_norm()
            << "        " << error_residual_norm.norm << "  " << error_residual_norm.u
            << "  " << error_residual_norm.p_fluid
            << "        " << error_update_norm.norm  << "  " << error_update_norm.u
            << "  " << error_update_norm.p_fluid
            << "  " << std::endl;

            // Update
            solution_delta_OUT += newton_update;
            newton_update = 0.0;
            newton_iteration++;
          }

        //If maximum allowed number of iterations for Newton algorithm are reached, print non-convergence message and abort program
        AssertThrow (newton_iteration < parameters.max_iterations_NR, ExcMessage("No convergence in nonlinear solver!"));
    }

    //Prints the header for convergence info on console
    template <int dim>
    void Solid<dim>::print_conv_header()
    {
        static const unsigned int l_width = 120;

        for (unsigned int i = 0; i < l_width; ++i)
          {
            pcout       << "_";
            outfile   << "_";
          }

        pcout       << std::endl;
        outfile   << std::endl;

        pcout   << "\n       SOLVER STEP      |    SYS_RES         "
                << "RES_NORM     RES_U      RES_P           "
                << "NU_NORM     NU_U       NU_P " << std::endl;
        outfile << "\n       SOLVER STEP      |    SYS_RES         "
                << "RES_NORM     RES_U      RES_P           "
                << "NU_NORM     NU_U       NU_P " << std::endl;

        for (unsigned int i = 0; i < l_width; ++i)
          {
            pcout     << "_";
            outfile   << "_";
          }
        pcout     << std::endl << std::endl;
        outfile   << std::endl << std::endl;
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
        pcout     << std::endl << std::endl;
        outfile   << std::endl << std::endl;

        pcout       << "Relative errors:" << std::endl
            << "Displacement:  " << error_update.u / error_update_0.u << std::endl
            << "Force (displ): " << error_residual.u / error_residual_0.u << std::endl
            << "Pore pressure: " << error_update.p_fluid / error_update_0.p_fluid << std::endl
            << "Force (pore):  " << error_residual.p_fluid / error_residual_0.p_fluid << std::endl;
        outfile   << "Relative errors:" << std::endl
            << "Displacement:  " << error_update.u / error_update_0.u << std::endl
            << "Force (displ): " << error_residual.u / error_residual_0.u << std::endl
            << "Pore pressure: " << error_update.p_fluid / error_update_0.p_fluid << std::endl
            << "Force (pore):  " << error_residual.p_fluid / error_residual_0.p_fluid << std::endl;
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
    void Solid<dim>::get_error_update(const TrilinosWrappers::MPI::BlockVector &newton_update_IN,
                                      Errors &error_update_OUT)
    {
        TrilinosWrappers::MPI::BlockVector error_ud(newton_update_IN);
        constraints.set_zero(error_ud);

        error_update_OUT.norm = error_ud.l2_norm();
        error_update_OUT.u = error_ud.block(u_block).l2_norm();
        error_update_OUT.p_fluid = error_ud.block(p_fluid_block).l2_norm();
    }

    //Compute the total solution, which is valid at any Newton step. This is required as, to reduce
    //computational error, the total solution is only updated at the end of the timestep.
    template <int dim>
    TrilinosWrappers::MPI::BlockVector
    Solid<dim>::get_total_solution(const TrilinosWrappers::MPI::BlockVector &solution_delta_IN) const
    {
        // Cell interpolation -> Ghosted vector
        TrilinosWrappers::MPI::BlockVector solution_total (locally_owned_partitioning,
                                                           locally_relevant_partitioning,
                                                           mpi_communicator,
                                                           /*vector_writable = */ false);
        TrilinosWrappers::MPI::BlockVector tmp (solution_total);
        solution_total = solution_n;
        tmp = solution_delta_IN;
        solution_total += tmp;
        return solution_total;
    }

    //Compute elemental stiffness tensor and right-hand side force vector, and assemble into global ones
    template <int dim>
    void Solid<dim>::assemble_system( const TrilinosWrappers::MPI::BlockVector &solution_delta )
    {
        timerconsole.enter_subsection("Assemble system");
        timerfile.enter_subsection("Assemble system");
        pcout       << " ASM_SYS " << std::flush;
        outfile   << " ASM_SYS " << std::flush;

        const TrilinosWrappers::MPI::BlockVector solution_total(get_total_solution(solution_delta));

        //Info given to FEValues and FEFaceValues constructors, to indicate which data will be needed at each element.
        const UpdateFlags uf_cell(update_values | update_gradients | update_JxW_values);
        const UpdateFlags uf_face(update_values | update_gradients | update_normal_vectors | update_quadrature_points | update_JxW_values );

        //Setup a copy of the data structures required for the process and pass them, along with the
        //memory addresses of the assembly functions to the WorkStream object for processing
        PerTaskData_ASM per_task_data(dofs_per_cell);
        ScratchData_ASM<ADNumberType> scratch_data(fe, qf_cell, uf_cell,  qf_face, uf_face, solution_total);

        FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
        cell (IteratorFilters::SubdomainEqualTo(this_mpi_process), dof_handler_ref.begin_active()),
        endc (IteratorFilters::SubdomainEqualTo(this_mpi_process), dof_handler_ref.end());
        for (; cell != endc; ++cell)
          {
            Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
            assemble_system_one_cell(cell, scratch_data, per_task_data); //, newton_iter_IN);
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
    void Solid<dim>::assemble_system_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                              ScratchData_ASM<ADNumberType> &scratch,
                                              PerTaskData_ASM &data) const
    {
        Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());

        data.reset();
        scratch.reset();
        scratch.fe_values_ref.reinit(cell);
        cell->get_dof_indices(data.local_dof_indices);

        // Setup automatic differentiation
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            // Initialise the dofs for the cell using the current solution.
            scratch.local_dof_values[k] = scratch.solution_total[data.local_dof_indices[k]];
            // Mark this cell DoF as an independent variable
            scratch.local_dof_values[k].diff(k, dofs_per_cell);
          }

        // Update the quadrature point solution
        // Compute the values and gradients of the solution in terms of the AD variables
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                const unsigned int k_group = fe.system_to_base_index(k).first.first;
                if (k_group == u_block)
                  {
                    const Tensor<2, dim> Grad_Nx_u = scratch.fe_values_ref[u_fe].gradient(k, q);
                    for (unsigned int dd = 0; dd < dim; dd++)
                      {
                        for (unsigned int ee = 0; ee < dim; ee++)
                          {
                            scratch.solution_grads_u_total[q][dd][ee] += scratch.local_dof_values[k] * Grad_Nx_u[dd][ee];
                          }
                      }
                  }
                else if  (k_group == p_fluid_block)
                  {
                    const double Nx_p = scratch.fe_values_ref[p_fluid_fe].value(k, q);
                    const Tensor<1, dim> Grad_Nx_p = scratch.fe_values_ref[p_fluid_fe].gradient(k, q);

                    scratch.solution_values_p_fluid_total[q] += scratch.local_dof_values[k] * Nx_p;
                    for (unsigned int dd = 0; dd < dim; dd++)
                      {
                        scratch.solution_grads_p_fluid_total[q][dd] += scratch.local_dof_values[k] * Grad_Nx_p[dd];
                      }
                  }
                else
                  Assert(k_group <= p_fluid_block, ExcInternalError());

              }
          }

        //Set up pointer "lgph" to the PointHistory object of this element
        const std::vector<std::shared_ptr<const PointHistory<dim, ADNumberType> > >
            lqph = quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());


        //Precalculate the element shape function values and gradients
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            Tensor<2, dim, ADNumberType> F_AD = scratch.solution_grads_u_total[q_point];
            F_AD += Tensor<2, dim, double>(Physics::Elasticity::StandardTensors<dim>::I);
            Assert(determinant(F_AD) > 0, ExcMessage("Invalid deformation map"));
            const Tensor<2, dim, ADNumberType> F_inv_AD = invert(F_AD);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int i_group = fe.system_to_base_index(i).first.first;

                if (i_group == u_block)
                  {
                    scratch.Nx[q_point][i] = scratch.fe_values_ref[u_fe].value(i, q_point);
                    scratch.grad_Nx[q_point][i] = scratch.fe_values_ref[u_fe].gradient(i, q_point)*F_inv_AD;
                    scratch.symm_grad_Nx[q_point][i] = symmetrize(scratch.grad_Nx[q_point][i]);
                  }
                else if  (i_group == p_fluid_block)
                  {
                    scratch.Nx_p_fluid[q_point][i] = scratch.fe_values_ref[p_fluid_fe].value(i, q_point);
                    scratch.grad_Nx_p_fluid[q_point][i] = scratch.fe_values_ref[p_fluid_fe].gradient(i, q_point)*F_inv_AD;
                  }
                else
                  Assert(i_group <= p_fluid_block, ExcInternalError());
              }
          }

        //Assemble the stiffness matrix and rhs vector
        std::vector<ADNumberType> residual_ad (dofs_per_cell, ADNumberType(0.0));
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            Tensor<2, dim, ADNumberType> F_AD = scratch.solution_grads_u_total[q_point];
            F_AD += Tensor<2, dim,double>(Physics::Elasticity::StandardTensors<dim>::I);
            const ADNumberType det_F_AD = determinant(F_AD);

            Assert(det_F_AD > 0, ExcInternalError());
            const Tensor<2, dim, ADNumberType> F_inv_AD = invert(F_AD); //inverse of def. gradient tensor

            const ADNumberType p_fluid = scratch.solution_values_p_fluid_total[q_point];

            {
              PointHistory<dim, ADNumberType> *lqph_q_point_nc = const_cast<PointHistory<dim, ADNumberType>*>(lqph[q_point].get());
              lqph_q_point_nc->update_internal_equilibrium(F_AD);
            }

            //Growth
            const Tensor<2, dim> Fg = lqph[q_point]->get_non_converged_growth_tensor();
            const Tensor<2, dim> Fg_inv = invert(Fg); //inverse of growth tensor
            const Tensor<2, dim, ADNumberType> Fve_AD = F_AD * Fg_inv;
            const ADNumberType det_Fve_AD = determinant(Fve_AD);  //Determinant of (visco)elastic part of def. gradient tensor
            Assert(det_Fve_AD > 0, ExcInternalError());

            //Get some info from constitutive model of solid
            static const SymmetricTensor< 2, dim, double> I (Physics::Elasticity::StandardTensors<dim>::I);
            const SymmetricTensor<2, dim, ADNumberType> tau_E = lqph[q_point]->get_tau_E(F_AD);
            SymmetricTensor<2, dim, ADNumberType> tau_fluid_vol (I);
            tau_fluid_vol *= -1.0 * p_fluid * det_F_AD;

            //Get some info from constitutive model of fluid
            const ADNumberType det_Fve_aux =  lqph[q_point]->get_converged_det_Fve();
            const double det_Fve_converged = Tensor<0,dim,double>(det_Fve_aux); //Needs to be double, not AD number
            const Tensor<1, dim, ADNumberType> overall_body_force = lqph[q_point]->get_overall_body_force(F_AD, parameters);

            // Define some aliases to make the assembly process easier to follow
            const std::vector<Tensor<1,dim>> &Nu = scratch.Nx[q_point];
            const std::vector<SymmetricTensor<2, dim, ADNumberType>> &symm_grad_Nu = scratch.symm_grad_Nx[q_point];
            const std::vector<double> &Np = scratch.Nx_p_fluid[q_point];
            const std::vector<Tensor<1, dim, ADNumberType> > &grad_Np = scratch.grad_Nx_p_fluid[q_point];
            const Tensor<1, dim, ADNumberType> grad_p = scratch.solution_grads_p_fluid_total[q_point]*F_inv_AD;
            const double JxW = scratch.fe_values_ref.JxW(q_point);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int i_group = fe.system_to_base_index(i).first.first;

                if (i_group == u_block)
                  {
                    residual_ad[i] += symm_grad_Nu[i] * ( tau_E + tau_fluid_vol ) * JxW;
                    residual_ad[i] -= Nu[i] * overall_body_force * JxW;
                  }
                else if (i_group == p_fluid_block)
                  {
                    const Tensor<1, dim, ADNumberType> seepage_vel_current = lqph[q_point]->get_seepage_velocity_current(F_AD, grad_p);
                    residual_ad[i] += Np[i] * (det_Fve_AD - det_Fve_converged) * JxW;
                    residual_ad[i] -= time.get_delta_t() * grad_Np[i] * seepage_vel_current * JxW;
                  }
                else
                  Assert(i_group <= p_fluid_block, ExcInternalError());
              }
          }

          // Assemble the Neumann contribution (external force contribution).
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) //Loop over faces in element
            {
              if (cell->face(face)->at_boundary() == true)
                {
                  scratch.fe_face_values_ref.reinit(cell, face);

                  for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point)
                    {
                      const Tensor<1, dim> &N = scratch.fe_face_values_ref.normal_vector(f_q_point);
                      const Point<dim>     &pt = scratch.fe_face_values_ref.quadrature_point(f_q_point);
                      const Tensor<1, dim> traction = get_neumann_traction(cell->face(face)->boundary_id(), pt, N);
                      const double flow = get_prescribed_fluid_flow(cell->face(face)->boundary_id(), pt);

                      if ( (traction.norm() < 1e-12) && (std::abs(flow) < 1e-12) ) continue;

                      const double JxW_f = scratch.fe_face_values_ref.JxW(f_q_point);

                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          const unsigned int i_group = fe.system_to_base_index(i).first.first;

                          if ((i_group == u_block) && (traction.norm() > 1e-12))
                          {
                              const unsigned int component_i = fe.system_to_component_index(i).first;
                              const double Nu_f = scratch.fe_face_values_ref.shape_value(i, f_q_point);
                              residual_ad[i] -= (Nu_f * traction[component_i]) * JxW_f;
                          }
                          if ((i_group == p_fluid_block) && (std::abs(flow) > 1e-12))
                          {
                              const double Nu_p = scratch.fe_face_values_ref.shape_value(i, f_q_point);
                              residual_ad[i] -= (Nu_p * flow) * JxW_f;
                          }
                        }
                    }
                }
            }

        // Linearise the residual
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
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
          cell (IteratorFilters::SubdomainEqualTo(this_mpi_process), dof_handler_ref.begin_active()),
          endc (IteratorFilters::SubdomainEqualTo(this_mpi_process), dof_handler_ref.end());
          for (; cell!=endc; ++cell)
          {
            Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
            const std::vector<std::shared_ptr<PointHistory<dim, ADNumberType> > >
                lqph = quadrature_point_history.get_data(cell);
            Assert(lqph.size() == n_q_points, ExcInternalError());
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              lqph[q_point]->update_end_timestep();
          }
    }


     //Solve the linearized equations
     template <int dim>
     void Solid<dim>::solve_linear_system( TrilinosWrappers::MPI::BlockVector &newton_update_OUT)
     {

           timerconsole.enter_subsection("Linear solver");
           timerfile.enter_subsection("Linear solver");
           pcout     << " SLV " << std::flush;
           outfile   << " SLV " << std::flush;

           TrilinosWrappers::MPI::Vector newton_update_nb;
           newton_update_nb.reinit(locally_owned_dofs, mpi_communicator);

           SolverControl solver_control (tangent_matrix_nb.m(), 1e-6 * system_rhs_nb.l2_norm());
           TrilinosWrappers::SolverDirect solver (solver_control);
           solver.solve(tangent_matrix_nb, newton_update_nb, system_rhs_nb);

           // Copy the non-block solution back to block system
           for (unsigned int i=0; i<locally_owned_dofs.n_elements(); ++i)
             {
               const types::global_dof_index idx_i = locally_owned_dofs.nth_index_in_set(i);
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
          first_cell ()
          {
            typename DataOut<dim, DH>::active_cell_iterator
            cell = this->dofs->begin_active();
            while ((cell != this->dofs->end()) && (cell->subdomain_id() != subdomain_id))
              ++cell;
            return cell;
          }

          virtual typename DataOut<dim, DH>::cell_iterator
          next_cell (const typename DataOut<dim, DH>::cell_iterator &old_cell)
          {
            if (old_cell != this->dofs->end())
              {
                const IteratorFilters::SubdomainEqualTo predicate(subdomain_id);
                return
                  ++(FilteredIterator<typename DataOut<dim, DH>::active_cell_iterator>
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
            DataPostprocessorVector<dim> ("grad_p",
                                          update_gradients),
            p_fluid_component (p_fluid_component)
          {}

          virtual ~GradientPostprocessor(){}

          virtual void
          evaluate_vector_field (const DataPostprocessorInputs::Vector<dim> &input_data,
                                 std::vector<Vector<double> >               &computed_quantities) const
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

    //Print results to file
    template <int dim>
    void Solid<dim>::output_results(const unsigned int timestep,
                                    const double current_time,
                                    TrilinosWrappers::MPI::BlockVector solution_IN,
                                    std::vector<Point<dim> > &tracked_vertices_IN,
                                    std::ofstream &plotpointfile) const
    {
            TrilinosWrappers::MPI::BlockVector solution_total ( locally_owned_partitioning,
                                                                locally_relevant_partitioning,
                                                                mpi_communicator,
                                                                false);
            solution_total = solution_IN;
            Vector<double> material_id;
            Vector<double> polynomial_order;
            material_id.reinit(triangulation.n_active_cells());
            polynomial_order.reinit(triangulation.n_active_cells());
            std::vector<types::subdomain_id> partition_int (triangulation.n_active_cells());
            GradientPostprocessor<dim> gradient_postprocessor (p_fluid_component);
            FilteredDataOut<dim> data_out(this_mpi_process);
            std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(dim,
                                          DataComponentInterpretation::component_is_part_of_vector);
            data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

            GridTools::get_subdomain_association (triangulation, partition_int);

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
            std::vector<Point<dim+1> > solution_vertices(tracked_vertices_IN.size());

            //Auxiliar variables needed for mpi processing
            Tensor<1, dim> sum_reaction_mpi;
            Tensor<1, dim> sum_reaction_pressure_mpi;
            Tensor<1, dim> sum_reaction_extra_mpi;
            sum_reaction_mpi = 0.0;
            sum_reaction_pressure_mpi = 0.0;
            sum_reaction_extra_mpi = 0.0;
            double sum_total_flow_mpi = 0.0;
            double sum_porous_dissipation_mpi = 0.0;
            double sum_viscous_dissipation_mpi = 0.0;
            double sum_solid_vol_mpi = 0.0;
            double sum_vol_current_mpi = 0.0;
            double sum_vol_reference_mpi = 0.0;

             //Declare local variables with number of stress components & assign value according to "dim" value
             unsigned int num_comp_symm_tensor = 6;

            //Declare local vectors to store values
            std::vector<Vector<double> > cauchy_stresses_total_elements (num_comp_symm_tensor,
                                                                         Vector<double> (triangulation.n_active_cells()));
            std::vector<Vector<double> > cauchy_stresses_E_elements (num_comp_symm_tensor,
                                                                     Vector<double> (triangulation.n_active_cells()));
            std::vector<Vector<double> > cauchy_stresses_p_elements (num_comp_symm_tensor,
                                                                     Vector<double> (triangulation.n_active_cells()));
            std::vector<Vector<double> > stretches_elements (dim,
                                                             Vector<double> (triangulation.n_active_cells()));
            std::vector<Vector<double> > seepage_velocity_elements (dim,
                                                                    Vector<double> (triangulation.n_active_cells()));
            Vector<double >porous_dissipation_elements (triangulation.n_active_cells());
            Vector<double >viscous_dissipation_elements (triangulation.n_active_cells());
            Vector<double >solid_vol_fraction_elements (triangulation.n_active_cells());

            Vector<double> growth_stretch_elements (triangulation.n_active_cells());

            //Declare and initialize local unit vectors (to construct tensor basis)
            std::vector<Tensor<1,dim> > basis_vectors (dim, Tensor<1,dim>() );
            for (unsigned int i=0; i<dim; ++i)
            {
                basis_vectors[i][i] = 1;
            }

            //Declare an instance of the material class object
            if (parameters.mat_type == "Neo-Hooke")
                NeoHooke<dim, ADNumberType> material( parameters.solid_vol_frac,
                                                      parameters.lambda,
                                                      parameters.growth_type,
                                                      parameters.growth_incr,
                                                      time,
                                                      parameters.eigen_solver,
                                                      parameters.mu );
            else if (parameters.mat_type == "Ogden")
                Ogden<dim, ADNumberType> material( parameters.solid_vol_frac,
                                                   parameters.lambda,
                                                   parameters.growth_type,
                                                   parameters.growth_incr,
                                                   time,
                                                   parameters.eigen_solver,
                                                   parameters.mu1_infty,
                                                   parameters.mu2_infty,
                                                   parameters.mu3_infty,
                                                   parameters.alpha1_infty,
                                                   parameters.alpha2_infty,
                                                   parameters.alpha3_infty );
            else if (parameters.mat_type == "visco-Ogden")
                visco_Ogden <dim, ADNumberType>material( parameters.solid_vol_frac,
                                                         parameters.lambda,
                                                         parameters.growth_type,
                                                         parameters.growth_incr,
                                                         time,
                                                         parameters.eigen_solver,
                                                         parameters.mu1_infty,
                                                         parameters.mu2_infty,
                                                         parameters.mu3_infty,
                                                         parameters.alpha1_infty,
                                                         parameters.alpha2_infty,
                                                         parameters.alpha3_infty,
                                                         parameters.mu1_mode_1,
                                                         parameters.mu2_mode_1,
                                                         parameters.mu3_mode_1,
                                                         parameters.alpha1_mode_1,
                                                         parameters.alpha2_mode_1,
                                                         parameters.alpha3_mode_1,
                                                         parameters.viscosity_mode_1);
            else
                Assert (false, ExcMessage("Material type not implemented"));

            //Define a local instance of FEValues to compute updated values required to calculate stresses
            const UpdateFlags uf_cell(update_values | update_gradients | update_JxW_values);
            FEValues<dim> fe_values_ref (fe, qf_cell, uf_cell);

            //Iterate through elements (cells) and Gauss Points
            FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
              cell (IteratorFilters::SubdomainEqualTo(this_mpi_process), dof_handler_ref.begin_active()),
              endc (IteratorFilters::SubdomainEqualTo(this_mpi_process), dof_handler_ref.end());
            for (; cell!=endc; ++cell)
            {
                if (cell->subdomain_id() != this_mpi_process) continue;
                material_id(cell->active_cell_index()) = static_cast<int>(cell->material_id());

                fe_values_ref.reinit(cell);

                std::vector<Tensor<2,dim> > solution_grads_u(n_q_points);
                fe_values_ref[u_fe].get_function_gradients(solution_total,solution_grads_u);

                std::vector< double > solution_values_p_fluid_total(n_q_points);
                fe_values_ref[p_fluid_fe].get_function_values(solution_total,solution_values_p_fluid_total);

                std::vector<Tensor<1,dim > > solution_grads_p_fluid_AD (n_q_points);
                fe_values_ref[p_fluid_fe].get_function_gradients(solution_total, solution_grads_p_fluid_AD);

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    const Tensor<2, dim, ADNumberType> F_AD = Physics::Elasticity::Kinematics::F(solution_grads_u[q_point]);
                    Tensor<2, dim> F;

                    const std::vector<std::shared_ptr<const PointHistory<dim, ADNumberType> > >
                        lqph = quadrature_point_history.get_data(cell);
                    Assert(lqph.size() == n_q_points, ExcInternalError());

                    const double p_fluid = solution_values_p_fluid_total[q_point];
                    double JxW = fe_values_ref.JxW(q_point);

                    //Growth
                    const Tensor<2, dim> Fg = lqph[q_point]->get_non_converged_growth_tensor();
                   // const Tensor<2, dim> Fg_inv = invert(Fg); //inverse of growth tensor
                   // const Tensor<2, dim, ADNumberType> Fve_AD = F_AD * Fg_inv;

                    //Cauchy stress
                    static const SymmetricTensor< 2, dim, double> I (Physics::Elasticity::StandardTensors<dim>::I);
                    SymmetricTensor<2, dim> sigma_E;
                    const SymmetricTensor<2, dim, ADNumberType> sigma_E_AD = lqph[q_point]->get_Cauchy_E(F_AD);

                    for (unsigned int i=0; i<dim; ++i)
                        for (unsigned int j=0; j<dim; ++j)
                        {
                           sigma_E[i][j] = Tensor<0,dim,double>(sigma_E_AD[i][j]);
                           F[i][j] = Tensor<0,dim,double>(F_AD[i][j]);
                        }

                    SymmetricTensor<2, dim> sigma_fluid_vol (I);
                    sigma_fluid_vol *= -p_fluid;
                    const SymmetricTensor<2, dim> sigma = sigma_E + sigma_fluid_vol;

                    //Volumes
                    const double det_Fg = determinant(Fg);
                    const double det_F = determinant(F);
                    sum_vol_current_mpi  += det_F * JxW;
                    sum_vol_reference_mpi += JxW;
                    const double solid_vol_fraction = parameters.solid_vol_frac/det_F;
                    sum_solid_vol_mpi += parameters.solid_vol_frac * JxW * det_Fg;

                    //Green-Lagrange strain
                    const Tensor<2, dim> E = 0.5 * (transpose(F)*F - I);

                    //Seepage velocity
                    const Tensor<2, dim, ADNumberType> F_inv = invert(F);
                    const Tensor<1,dim, ADNumberType > grad_p_fluid_AD =  solution_grads_p_fluid_AD[q_point]*F_inv;
                    const Tensor<1, dim, ADNumberType> seepage_vel_AD = lqph[q_point]->get_seepage_velocity_current(F_AD, grad_p_fluid_AD);

                    //Dissipations
                    const double porous_dissipation = lqph[q_point]->get_porous_dissipation(F_AD, grad_p_fluid_AD);
                    sum_porous_dissipation_mpi += porous_dissipation * det_F * JxW;

                    const double viscous_dissipation = lqph[q_point]->get_viscous_dissipation();
                    sum_viscous_dissipation_mpi += viscous_dissipation * det_F * JxW;

                    //Growth
                    const double growth_stretch = lqph[q_point]->get_converged_growth_stretch();

                    for (unsigned int j=0; j<dim; ++j)
                    {
                        cauchy_stresses_total_elements[j][cell->active_cell_index()] += ((sigma * basis_vectors[j])* basis_vectors[j])/n_q_points;
                        cauchy_stresses_E_elements[j][cell->active_cell_index()] += ((sigma_E * basis_vectors[j])* basis_vectors[j])/n_q_points;
                        cauchy_stresses_p_elements[j][cell->active_cell_index()] += ((sigma_fluid_vol * basis_vectors[j])* basis_vectors[j])/n_q_points;
                        stretches_elements[j][cell->active_cell_index()] += std::sqrt(1.0+2.0*E[j][j])/n_q_points;
                        seepage_velocity_elements[j][cell->active_cell_index()] +=  Tensor<0,dim,double>(seepage_vel_AD[j])/n_q_points;
                    }

                    porous_dissipation_elements[cell->active_cell_index()] +=  porous_dissipation/n_q_points;
                    viscous_dissipation_elements[cell->active_cell_index()] +=  viscous_dissipation/n_q_points;
                    solid_vol_fraction_elements[cell->active_cell_index()] +=  solid_vol_fraction/n_q_points;

                    growth_stretch_elements[cell->active_cell_index()] += growth_stretch/n_q_points;

                    cauchy_stresses_total_elements[3][cell->active_cell_index()] += ((sigma * basis_vectors[0])* basis_vectors[1])/n_q_points;
                    cauchy_stresses_total_elements[4][cell->active_cell_index()] += ((sigma * basis_vectors[0])* basis_vectors[2])/n_q_points;

                    cauchy_stresses_E_elements[3][cell->active_cell_index()] += ((sigma_E * basis_vectors[0])* basis_vectors[1])/n_q_points;
                    cauchy_stresses_E_elements[4][cell->active_cell_index()] += ((sigma_E * basis_vectors[0])* basis_vectors[2])/n_q_points;

                    cauchy_stresses_p_elements[3][cell->active_cell_index()] += ((sigma_fluid_vol * basis_vectors[0])* basis_vectors[1])/n_q_points;
                    cauchy_stresses_p_elements[4][cell->active_cell_index()] += ((sigma_fluid_vol * basis_vectors[0])* basis_vectors[2])/n_q_points;
                }

       // Compute reaction force on load boundary & total fluid flow across drained boundary
       // Define a local instance of FEFaceValues to compute values required to calculate reaction force
                const UpdateFlags uf_face( update_values | update_gradients | update_normal_vectors | update_JxW_values );
                FEFaceValues<dim> fe_face_values_ref(fe, qf_face, uf_face);

                for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                {
                    //Reaction force
                    if (cell->face(face)->at_boundary() == true &&
                        cell->face(face)->boundary_id() == get_reaction_boundary_id_for_output())
                    {
                        fe_face_values_ref.reinit(cell, face);

                        //Get displacement gradients for current face
                        std::vector<Tensor<2,dim> > solution_grads_u_f(n_q_points_f);
                        fe_face_values_ref[u_fe].get_function_gradients(solution_total,solution_grads_u_f);

                        //Get pressure for current element
                        std::vector< double > solution_values_p_fluid_total_f(n_q_points_f);
                        fe_face_values_ref[p_fluid_fe].get_function_values(solution_total,solution_values_p_fluid_total_f);

                        for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point)
                        {
                            const Tensor<1, dim> &N = fe_face_values_ref.normal_vector(f_q_point);
                            const double JxW_f = fe_face_values_ref.JxW(f_q_point);

                            //Compute deformation gradient from displacements gradient (present configuration)
                            const Tensor<2, dim, ADNumberType> F_AD = Physics::Elasticity::Kinematics::F(solution_grads_u_f[f_q_point]);

                            const std::vector<std::shared_ptr<const PointHistory<dim, ADNumberType> > >
                                lqph = quadrature_point_history.get_data(cell);
                            Assert(lqph.size() == n_q_points, ExcInternalError());

                            const double p_fluid = solution_values_p_fluid_total[f_q_point];

                            //Cauchy stress
                            static const SymmetricTensor< 2, dim, double> I (Physics::Elasticity::StandardTensors<dim>::I);
                            SymmetricTensor<2, dim> sigma_E;
                            const SymmetricTensor<2, dim, ADNumberType> sigma_E_AD = lqph[f_q_point]->get_Cauchy_E(F_AD);

                            for (unsigned int i=0; i<dim; ++i)
                                for (unsigned int j=0; j<dim; ++j)
                                {
                                   sigma_E[i][j] = Tensor<0,dim,double>(sigma_E_AD[i][j]);
                                }

                            SymmetricTensor<2, dim> sigma_fluid_vol (I);
                            sigma_fluid_vol *= -1.0*p_fluid;
                            const SymmetricTensor<2, dim> sigma = sigma_E + sigma_fluid_vol;
                            sum_reaction_mpi += sigma * N * JxW_f;
                            sum_reaction_pressure_mpi += sigma_fluid_vol * N * JxW_f;
                            sum_reaction_extra_mpi += sigma_E * N * JxW_f;
                        }
                    }

                    //Fluid flow
                    if (cell->face(face)->at_boundary() == true &&
                       ( cell->face(face)->boundary_id() == get_drained_boundary_id_for_output().first ||
                         cell->face(face)->boundary_id() == get_drained_boundary_id_for_output().second ) )
                    {
                        fe_face_values_ref.reinit(cell, face);

                        //Get displacement gradients for current face
                        std::vector<Tensor<2,dim> > solution_grads_u_f(n_q_points_f);
                        fe_face_values_ref[u_fe].get_function_gradients(solution_total,solution_grads_u_f);

                        //Get pressure gradients for current face
                        std::vector<Tensor<1,dim> > solution_grads_p_f(n_q_points_f);
                        fe_face_values_ref[p_fluid_fe].get_function_gradients(solution_total,solution_grads_p_f);


                        for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point)
                        {
                            const Tensor<1, dim> &N = fe_face_values_ref.normal_vector(f_q_point);
                            const double JxW_f = fe_face_values_ref.JxW(f_q_point);

                            //Deformation gradient and inverse from displacements gradient (present configuration)
                            const Tensor<2, dim, ADNumberType> F_AD = Physics::Elasticity::Kinematics::F(solution_grads_u_f[f_q_point]);

                            const Tensor<2, dim, ADNumberType> F_inv_AD = invert(F_AD);
                            ADNumberType det_F_AD = determinant(F_AD);

                            const std::vector<std::shared_ptr<const PointHistory<dim, ADNumberType> > >
                                lqph = quadrature_point_history.get_data(cell);
                            Assert(lqph.size() == n_q_points, ExcInternalError());

                            //Seepage velocity
                            Tensor<1, dim> seepage;
                            double det_F = Tensor<0,dim,double>(det_F_AD);
                            const Tensor<1, dim, ADNumberType> grad_p = solution_grads_p_f[f_q_point]*F_inv_AD;
                            const Tensor<1, dim, ADNumberType> seepage_AD
                                   = lqph[f_q_point]->get_seepage_velocity_current(F_AD, grad_p);

                            for (unsigned int i=0; i<dim; ++i)
                                seepage[i] = Tensor<0,dim,double>(seepage_AD[i]);

                            sum_total_flow_mpi += (seepage/det_F) * N * JxW_f;
                        }
                    }
                }
            }

            //Sum the results from different MPI process and then add to the reaction_force vector
            //In theory, the solution on each surface (each cell) only exists in one MPI process
            //so, we add all MPI process, one will have the solution and the others will be zero
            for (unsigned int d=0; d<dim; ++d)
            {
                reaction_force[d] = Utilities::MPI::sum(sum_reaction_mpi[d], mpi_communicator);
                reaction_force_pressure[d] = Utilities::MPI::sum(sum_reaction_pressure_mpi[d], mpi_communicator);
                reaction_force_extra[d] = Utilities::MPI::sum(sum_reaction_extra_mpi[d], mpi_communicator);
            }

            //Same for total fluid flow, and for porous and viscous dissipations
            total_fluid_flow = Utilities::MPI::sum(sum_total_flow_mpi, mpi_communicator);
            total_porous_dissipation = Utilities::MPI::sum(sum_porous_dissipation_mpi, mpi_communicator);
            total_viscous_dissipation = Utilities::MPI::sum(sum_viscous_dissipation_mpi, mpi_communicator);
            total_solid_vol = Utilities::MPI::sum(sum_solid_vol_mpi, mpi_communicator);
            total_vol_current = Utilities::MPI::sum(sum_vol_current_mpi, mpi_communicator);
            total_vol_reference = Utilities::MPI::sum(sum_vol_reference_mpi, mpi_communicator);

        //  Extract solution for tracked vectors
        // Copying an MPI::BlockVector into MPI::Vector is not possible, so we copy each block of MPI::BlockVector into an MPI::Vector
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
                Vector<double> solution_vector(solution_p_vector.size()+solution_u_vector.size());

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
                       if ( abs(update[d]) < 1.5*parameters.tol_u )
                           update[d] = 0.0;
                       solution_vertices[p][d] = update[d];
                   }
                }
        // Write the results to the plotting file.
        // Add two blank lines between cycles in the cyclic loading examples so GNUPLOT can detect each cycle as a different block
                if (( (parameters.geom_type == "Budday_cube_tension_compression_fully_fixed")||
                      (parameters.geom_type == "Budday_cube_tension_compression")||
                      (parameters.geom_type == "Budday_cube_shear_fully_fixed")                  ) &&
                    ( (abs(current_time - parameters.end_time/3.)    < 0.9*parameters.delta_t)||
                      (abs(current_time - 2.*parameters.end_time/3.) < 0.9*parameters.delta_t)   ) &&
                      parameters.num_cycle_sets == 1 )
                {
                    plotpointfile << std::endl<< std::endl;
                }
                if (( (parameters.geom_type == "Budday_cube_tension_compression_fully_fixed")||
                      (parameters.geom_type == "Budday_cube_tension_compression")||
                      (parameters.geom_type == "Budday_cube_shear_fully_fixed")             ) &&
                    ( (abs(current_time - parameters.end_time/9.)    < 0.9*parameters.delta_t)||
                      (abs(current_time - 2.*parameters.end_time/9.) < 0.9*parameters.delta_t)||
                      (abs(current_time - 3.*parameters.end_time/9.) < 0.9*parameters.delta_t)||
                      (abs(current_time - 5.*parameters.end_time/9.) < 0.9*parameters.delta_t)||
                      (abs(current_time - 7.*parameters.end_time/9.) < 0.9*parameters.delta_t) ) &&
                      parameters.num_cycle_sets == 2 )
                {
                    plotpointfile << std::endl<< std::endl;
                }

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

// Add the results to the solution to create the output file for Paraview
            std::vector<std::string> solution_name(dim, "displacement");
            solution_name.push_back("pore_pressure");

            data_out.attach_dof_handler(dof_handler_ref);
            data_out.add_data_vector(solution_total,
                                     solution_name,
                                     DataOut<dim>::type_dof_data,
                                     data_component_interpretation);

            data_out.add_data_vector (solution_total,
                                      gradient_postprocessor);

            const Vector<double> partitioning(partition_int.begin(),
                                              partition_int.end());

            data_out.add_data_vector (material_id, "material_id");
            data_out.add_data_vector (partitioning, "partitioning");

            data_out.add_data_vector (cauchy_stresses_total_elements[0], "sigma_total_xx");
            data_out.add_data_vector (cauchy_stresses_total_elements[1], "sigma_total_yy");
            data_out.add_data_vector (cauchy_stresses_total_elements[2], "sigma_total_zz");
            data_out.add_data_vector (cauchy_stresses_total_elements[3], "sigma_total_xy");
            data_out.add_data_vector (cauchy_stresses_total_elements[4], "sigma_total_xz");
            data_out.add_data_vector (cauchy_stresses_total_elements[5], "sigma_total_yz");

            data_out.add_data_vector (cauchy_stresses_E_elements[0], "sigma_extra_xx");
            data_out.add_data_vector (cauchy_stresses_E_elements[1], "sigma_extra_yy");
            data_out.add_data_vector (cauchy_stresses_E_elements[2], "sigma_extra_zz");
            data_out.add_data_vector (cauchy_stresses_E_elements[3], "sigma_extra_xy");
            data_out.add_data_vector (cauchy_stresses_E_elements[4], "sigma_extra_xz");
            data_out.add_data_vector (cauchy_stresses_E_elements[5], "sigma_extra_yz");

            data_out.add_data_vector (cauchy_stresses_p_elements[0], "sigma_pressure_xx");
            data_out.add_data_vector (cauchy_stresses_p_elements[1], "sigma_pressure_yy");
            data_out.add_data_vector (cauchy_stresses_p_elements[2], "sigma_pressure_zz");
            data_out.add_data_vector (cauchy_stresses_p_elements[3], "sigma_pressure_xy");
            data_out.add_data_vector (cauchy_stresses_p_elements[4], "sigma_pressure_xz");
            data_out.add_data_vector (cauchy_stresses_p_elements[5], "sigma_pressure_yz");

            data_out.add_data_vector (stretches_elements[0], "stretch_xx");
            data_out.add_data_vector (stretches_elements[1], "stretch_yy");
            data_out.add_data_vector (stretches_elements[2], "stretch_zz");

            data_out.add_data_vector (growth_stretch_elements, "growth_stretch");

            data_out.build_patches(degree_displ);
            struct Filename
            {
              static std::string get_filename_vtu (unsigned int process,
                                                   unsigned int timestep,
                                                   const unsigned int n_digits = 5)
              {
                std::ostringstream filename_vtu;
                filename_vtu
                << "solution."
                << Utilities::int_to_string (process, n_digits)
                << "."
                << Utilities::int_to_string(timestep, n_digits)
                << ".vtu";
                return filename_vtu.str();
              }

              static std::string get_filename_pvtu (unsigned int timestep,
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

            const std::string filename_vtu = Filename::get_filename_vtu(this_mpi_process, timestep);
            std::ofstream output(filename_vtu.c_str());
            data_out.write_vtu(output);

            // We have a collection of files written in parallel
            // This next set of steps should only be performed by master process
            if (this_mpi_process == 0)
            {
              // List of all files written out at this timestep by all processors
              std::vector<std::string> parallel_filenames_vtu;
              for (unsigned int p=0; p < n_mpi_processes; ++p)
              {
                parallel_filenames_vtu.push_back(Filename::get_filename_vtu(p, timestep));
              }

              const std::string filename_pvtu (Filename::get_filename_pvtu(timestep));
              std::ofstream pvtu_master(filename_pvtu.c_str());
              data_out.write_pvtu_record(pvtu_master,
                                         parallel_filenames_vtu);

              // Time dependent data master file
              static std::vector<std::pair<double,std::string> > time_and_name_history;
              time_and_name_history.push_back (std::make_pair (current_time,
                                                               filename_pvtu));
              const std::string filename_pvd (Filename::get_filename_pvd());
              std::ofstream pvd_output (filename_pvd.c_str());
              DataOutBase::write_pvd_record (pvd_output, time_and_name_history);
            }
    }

    //Header for console output file
    template <int dim>
    void Solid<dim>::print_console_file_header(std::ofstream &outputfile) const
    {
            outputfile << "/*-----------------------------------------------------------------------------------------";
            outputfile << "\n\n  Poro-viscoelastic formulation to solve nonlinear solid mechanics problems using deal.ii";
            outputfile << "\n\n  Problem setup by E Comellas and J-P Pelteret, University of Erlangen-Nuremberg, 2018";
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
                          << std::right << std::setw(16)<< "ref vol,"
                          << std::right << std::setw(16)<< "def vol,"
                          << std::right << std::setw(16)<< "solid vol,";
            for (unsigned int p=0; p<tracked_vertices.size(); ++p)
                for (unsigned int d=0; d<(dim+1); ++d)
                    plotpointfile << std::right<< std::setw(11) <<"P" << p << "[" << d << "],";

            for (unsigned int d=0; d<dim; ++d)
                plotpointfile << std::right<< std::setw(13) << "reaction [" << d << "],";

            for (unsigned int d=0; d<dim; ++d)
                plotpointfile << std::right<< std::setw(13) << "reac(p) [" << d << "],";

            for (unsigned int d=0; d<dim; ++d)
                plotpointfile << std::right<< std::setw(13) << "reac(E) [" << d << "],";

            plotpointfile << std::right<< std::setw(16)<< "fluid flow,"
                          << std::right<< std::setw(16)<< "porous dissip,"
                          << std::right<< std::setw(15)<< "viscous dissip"
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


    // @sect3{Validation examples from Ehlers and Eipper 1999}
    // We group the definition of the geometry, boundary and loading conditions specific to
    // the validation examples from Ehlers and Eipper 1999 into specific classes.

    //@sect4{Base class: Tube geometry and boundary conditions}
    template <int dim>
    class ValidationEhlers1999TubeBase
          : public Solid<dim>
    {
        public:
          ValidationEhlers1999TubeBase (const Parameters::AllParameters &parameters)
            : Solid<dim> (parameters)
          {}

          virtual ~ValidationEhlers1999TubeBase () {}

        private:
          virtual void make_grid()
          {
            GridGenerator::cylinder( this->triangulation,
                                     0.1,
                                     0.5);

            const double rot_angle = 3.0*numbers::PI/2.0;
            GridTools::rotate( rot_angle, 1, this->triangulation);

            this->triangulation.reset_manifold(0);
            static const CylindricalManifold<dim> manifold_description_3d(2);
            this->triangulation.set_manifold (0, manifold_description_3d);
            GridTools::scale(this->parameters.scale, this->triangulation);
            this->triangulation.refine_global(std::max (1U, this->parameters.global_refinement));
            this->triangulation.reset_manifold(0);
          }

          virtual void define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices)
          {
            tracked_vertices[0][0] = 0.0*this->parameters.scale;
            tracked_vertices[0][1] = 0.0*this->parameters.scale;
            tracked_vertices[0][2] = 0.5*this->parameters.scale;

            tracked_vertices[1][0] = 0.0*this->parameters.scale;
            tracked_vertices[1][1] = 0.0*this->parameters.scale;
            tracked_vertices[1][2] = -0.5*this->parameters.scale;
          }

          virtual void make_dirichlet_constraints(ConstraintMatrix &constraints)
          {
            if (this->time.get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       2,
                                                       ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
            }
            else
            {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       2,
                                                       ZeroFunction<dim>(this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
            }

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      0,
                                                      ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      (this->fe.component_mask(this->x_displacement)|
                                                       this->fe.component_mask(this->y_displacement)  ) );

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      1,
                                                      ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      (this->fe.component_mask(this->x_displacement) |
                                                       this->fe.component_mask(this->y_displacement) |
                                                       this->fe.component_mask(this->z_displacement) ));
          }

          virtual double
          get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                          const Point<dim>         &pt) const
          {
              (void)pt;
              (void)boundary_id;
              return 0.0;
          }

          virtual types::boundary_id
          get_reaction_boundary_id_for_output() const
          {
              return 2;
          }

          virtual  std::pair<types::boundary_id,types::boundary_id>
          get_drained_boundary_id_for_output() const
          {
              return std::make_pair(2,2);
          }

          virtual  std::pair<double, FEValuesExtractors::Scalar>
          get_dirichlet_load(const types::boundary_id &boundary_id) const
          {
              double displ_incr = 0;
              FEValuesExtractors::Scalar direction;

              (void)boundary_id;
              throw std::runtime_error ("Displacement loading not implemented for Ehlers validation examples.");

              return std::make_pair(displ_incr,direction);
          }
    };

    //@sect4{Derived class: Step load example}
    template <int dim>
    class ValidationEhlers1999StepLoad
          : public ValidationEhlers1999TubeBase<dim>
    {
        public:
          ValidationEhlers1999StepLoad (const Parameters::AllParameters &parameters)
            : ValidationEhlers1999TubeBase<dim> (parameters)
          {}

          virtual ~ValidationEhlers1999StepLoad () {}

        private:
            virtual Tensor<1,dim>
            get_neumann_traction (const types::boundary_id &boundary_id,
                                  const Point<dim>         &pt,
                                  const Tensor<1,dim>      &N) const
            {
              if (this->parameters.load_type == "pressure")
              {
                if (boundary_id == 2)
                {
                  return this->parameters.load * N;
                }
              }

              (void)pt;

              return Tensor<1,dim>();
            }
    };

    //@sect4{Derived class: Load increasing example}
    template <int dim>
    class ValidationEhlers1999IncreaseLoad
          : public ValidationEhlers1999TubeBase<dim>
    {
        public:
          ValidationEhlers1999IncreaseLoad (const Parameters::AllParameters &parameters)
            : ValidationEhlers1999TubeBase<dim> (parameters)
          {}

          virtual ~ValidationEhlers1999IncreaseLoad () {}

        private:
            virtual Tensor<1,dim>
            get_neumann_traction (const types::boundary_id &boundary_id,
                                  const Point<dim>         &pt,
                                  const Tensor<1,dim>      &N) const
            {
              if (this->parameters.load_type == "pressure")
              {
                if (boundary_id == 2)
                {
                  const double initial_load = this->parameters.load;
                  const double final_load = 20.0*initial_load;
                  const double initial_time = this->time.get_delta_t();
                  const double final_time = this->time.get_end();
                  const double current_time = this->time.get_current();
                  const double load = initial_load + (final_load-initial_load)*(current_time-initial_time)/(final_time-initial_time);
                  return load * N;
                }
              }

              (void)pt;

              return Tensor<1,dim>();
            }
    };

    //@sect4{Class: Consolidation cube}
    template <int dim>
    class ValidationEhlers1999CubeConsolidation
          : public Solid<dim>
    {
        public:
          ValidationEhlers1999CubeConsolidation (const Parameters::AllParameters &parameters)
            : Solid<dim> (parameters)
          {}

          virtual ~ValidationEhlers1999CubeConsolidation () {}

        private:
          virtual void
          make_grid()
          {
             GridGenerator::hyper_rectangle(this->triangulation,
                                            Point<dim>(0.0, 0.0, 0.0),
                                            Point<dim>(1.0, 1.0, 1.0),
                                            true);

             GridTools::scale(this->parameters.scale, this->triangulation);
             this->triangulation.refine_global(std::max (1U, this->parameters.global_refinement));

             typename Triangulation<dim>::active_cell_iterator cell =
                     this->triangulation.begin_active(), endc = this->triangulation.end();
             for (; cell != endc; ++cell)
             {
               for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                 if (cell->face(face)->at_boundary() == true  &&
                     cell->face(face)->center()[2] == 1.0 * this->parameters.scale)
                 {
                   if (cell->face(face)->center()[0] < 0.5 * this->parameters.scale  &&
                       cell->face(face)->center()[1] < 0.5 * this->parameters.scale)
                       cell->face(face)->set_boundary_id(100);
                   else
                       cell->face(face)->set_boundary_id(101);
                 }
             }
          }

          virtual void
          define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices)
          {
            tracked_vertices[0][0] = 0.0*this->parameters.scale;
            tracked_vertices[0][1] = 0.0*this->parameters.scale;
            tracked_vertices[0][2] = 1.0*this->parameters.scale;

            tracked_vertices[1][0] = 0.0*this->parameters.scale;
            tracked_vertices[1][1] = 0.0*this->parameters.scale;
            tracked_vertices[1][2] = 0.0*this->parameters.scale;
          }

          virtual void
          make_dirichlet_constraints(ConstraintMatrix &constraints)
          {
            if (this->time.get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       101,
                                                       ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
            }
            else
            {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       101,
                                                       ZeroFunction<dim>(this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
            }

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      0,
                                                      ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      this->fe.component_mask(this->x_displacement));

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      1,
                                                      ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      this->fe.component_mask(this->x_displacement));

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      2,
                                                      ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      this->fe.component_mask(this->y_displacement));

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      3,
                                                      ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      this->fe.component_mask(this->y_displacement));

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      4,
                                                      ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      ( this->fe.component_mask(this->x_displacement) |
                                                        this->fe.component_mask(this->y_displacement) |
                                                        this->fe.component_mask(this->z_displacement) ));
          }

          virtual Tensor<1,dim>
          get_neumann_traction (const types::boundary_id &boundary_id,
                                const Point<dim>         &pt,
                                const Tensor<1,dim>      &N) const
          {
            if (this->parameters.load_type == "pressure")
            {
              if (boundary_id == 100)
              {
                return this->parameters.load * N;
              }
            }

            (void)pt;

            return Tensor<1,dim>();
          }

          virtual double
          get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                          const Point<dim>         &pt) const
          {
              (void)pt;
              (void)boundary_id;
              return 0.0;
          }

          virtual types::boundary_id
          get_reaction_boundary_id_for_output() const
          {
              return 100;
          }

          virtual  std::pair<types::boundary_id,types::boundary_id>
          get_drained_boundary_id_for_output() const
          {
              return std::make_pair(101,101);
          }

          virtual  std::pair<double, FEValuesExtractors::Scalar>
          get_dirichlet_load(const types::boundary_id &boundary_id) const
          {
              double displ_incr = 0;
              FEValuesExtractors::Scalar direction;

              (void)boundary_id;
              throw std::runtime_error ("Displacement loading not implemented for Ehlers validation examples.");

              return std::make_pair(displ_incr,direction);
          }
    };

    // @sect3{Validation examples from Franceschini et al. 2006}
    // We group the definition of the geometry, boundary and loading conditions specific to
    // the validation examples from Franceschini et al. 2006 into a specific class.

    //@sect4{Franceschini experiments}
    template <int dim>
    class Franceschini2006Consolidation
          : public Solid<dim>
    {
        public:
        Franceschini2006Consolidation (const Parameters::AllParameters &parameters)
            : Solid<dim> (parameters)
          {}

          virtual ~Franceschini2006Consolidation () {}

        private:
          virtual void make_grid()
          {
            const Point<dim-1> mesh_center(0.0, 0.0);
            const double radius = 0.5;
            const double height = 0.23;
            Triangulation<dim-1> triangulation_in;
            GridGenerator::hyper_ball( triangulation_in,
                                       mesh_center,
                                       radius);

            GridGenerator::extrude_triangulation(triangulation_in,
                                                  2,
                                                  height,
                                                  this->triangulation);

            const CylindricalManifold<dim> cylinder_3d(2);
            const types::manifold_id cylinder_id = 0;


            this->triangulation.set_manifold(cylinder_id, cylinder_3d);

            for (auto cell : this->triangulation.active_cell_iterators())
            {
              for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
              {
                if (cell->face(face)->at_boundary() == true)
                {
                  if (cell->face(face)->center()[2] == 0.0)
                      cell->face(face)->set_boundary_id(1);

                  else if (cell->face(face)->center()[2] == height)
                      cell->face(face)->set_boundary_id(2);

                  else
                  {
                      cell->face(face)->set_boundary_id(0);
                      cell->face(face)->set_all_manifold_ids(cylinder_id);
                  }
                }
              }
            }

            GridTools::scale(this->parameters.scale, this->triangulation);
            this->triangulation.refine_global(std::max (1U, this->parameters.global_refinement));
          }

          virtual void define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices)
          {
            tracked_vertices[0][0] = 0.0*this->parameters.scale;
            tracked_vertices[0][1] = 0.0*this->parameters.scale;
            tracked_vertices[0][2] = 0.23*this->parameters.scale;

            tracked_vertices[1][0] = 0.0*this->parameters.scale;
            tracked_vertices[1][1] = 0.0*this->parameters.scale;
            tracked_vertices[1][2] = 0.0*this->parameters.scale;
          }

          virtual void make_dirichlet_constraints(ConstraintMatrix &constraints)
          {
            if (this->time.get_timestep() < 2)
            {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       1,
                                                       ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));

              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       2,
                                                       ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
            }
            else
            {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       1,
                                                       ZeroFunction<dim>(this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));

              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       2,
                                                       ZeroFunction<dim>(this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
            }

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      0,
                                                      ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      (this->fe.component_mask(this->x_displacement)|
                                                       this->fe.component_mask(this->y_displacement)  ) );

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      1,
                                                      ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      (this->fe.component_mask(this->x_displacement) |
                                                       this->fe.component_mask(this->y_displacement) |
                                                       this->fe.component_mask(this->z_displacement) ));

            VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                      2,
                                                      ZeroFunction<dim>(this->n_components),
                                                      constraints,
                                                      (this->fe.component_mask(this->x_displacement) |
                                                       this->fe.component_mask(this->y_displacement) ));
          }

          virtual double
          get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                                     const Point<dim>         &pt) const
          {
              (void)pt;
              (void)boundary_id;
              return 0.0;
          }

          virtual types::boundary_id
          get_reaction_boundary_id_for_output() const
          {
              return 2;
          }

          virtual  std::pair<types::boundary_id,types::boundary_id>
          get_drained_boundary_id_for_output() const
          {
              return std::make_pair(1,2);
          }

          virtual  std::pair<double, FEValuesExtractors::Scalar>
          get_dirichlet_load(const types::boundary_id &boundary_id) const
          {
              double displ_incr = 0;
              FEValuesExtractors::Scalar direction;

              (void)boundary_id;
              throw std::runtime_error ("Displacement loading not implemented for Franceschini examples.");

              return std::make_pair(displ_incr,direction);
          }

          virtual Tensor<1,dim>
          get_neumann_traction (const types::boundary_id &boundary_id,
                                const Point<dim>         &pt,
                                const Tensor<1,dim>      &N) const
          {
            if (this->parameters.load_type == "pressure")
            {
              if (boundary_id == 2)
              {
                return (this->parameters.load * N);


                const double final_load = this->parameters.load;
                const double final_load_time = 10 * this->time.get_delta_t();
                const double current_time = this->time.get_current();


                const double c = final_load_time / 2.0;
                const double r = 200.0 * 0.03 / c;

                const double load = final_load * std::exp(r * current_time)
                                    / ( std::exp(c * current_time) +  std::exp(r * current_time));
                return load * N;
              }
            }

            (void)pt;

            return Tensor<1,dim>();
          }
    };

    // @sect3{Examples to reproduce experiments by Budday et al. 2017}
    // We group the definition of the geometry, boundary and loading conditions specific to
    // the examples to reproduce experiments by Budday et al. 2017 into specific classes.

    //@sect4{Base class: Cube geometry and loading pattern}
    template <int dim>
    class BrainBudday2017BaseCube
          : public Solid<dim>
    {
        public:
            BrainBudday2017BaseCube (const Parameters::AllParameters &parameters)
            : Solid<dim> (parameters)
          {}

          virtual ~BrainBudday2017BaseCube () {}

        private:
          virtual void
          make_grid()
          {
            GridGenerator::hyper_cube(this->triangulation,
                                      0.0,
                                      1.0,
                                      true);

            typename Triangulation<dim>::active_cell_iterator cell =
                    this->triangulation.begin_active(), endc = this->triangulation.end();
            for (; cell != endc; ++cell)
            {
              for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                if (cell->face(face)->at_boundary() == true  &&
                    ( cell->face(face)->boundary_id() == 0 ||
                      cell->face(face)->boundary_id() == 1 ||
                      cell->face(face)->boundary_id() == 2 ||
                      cell->face(face)->boundary_id() == 3    ) )

                      cell->face(face)->set_boundary_id(100);

            }

            GridTools::scale(this->parameters.scale, this->triangulation);
            this->triangulation.refine_global(std::max (1U, this->parameters.global_refinement));
          }

          virtual double
          get_prescribed_fluid_flow (const types::boundary_id &boundary_id,
                          const Point<dim>         &pt) const
          {
              (void)pt;
              (void)boundary_id;
              return 0.0;
          }

          virtual  std::pair<types::boundary_id,types::boundary_id>
          get_drained_boundary_id_for_output() const
          {
              return std::make_pair(100,100);
          }
    };

    //@sect4{Derived class: Uniaxial boundary conditions}
    template <int dim>
    class BrainBudday2017CubeTensionCompression
          : public BrainBudday2017BaseCube<dim>
    {
        public:
          BrainBudday2017CubeTensionCompression (const Parameters::AllParameters &parameters)
            : BrainBudday2017BaseCube<dim> (parameters)
          {}

          virtual ~BrainBudday2017CubeTensionCompression () {}

        private:
          virtual void
          define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices)
          {
            tracked_vertices[0][0] = 0.5*this->parameters.scale;
            tracked_vertices[0][1] = 0.5*this->parameters.scale;
            tracked_vertices[0][2] = 1.0*this->parameters.scale;

            tracked_vertices[1][0] = 0.5*this->parameters.scale;
            tracked_vertices[1][1] = 0.5*this->parameters.scale;
            tracked_vertices[1][2] = 0.5*this->parameters.scale;
          }

          virtual void
          make_dirichlet_constraints(ConstraintMatrix &constraints)
          {
              if (this->time.get_timestep() < 2)
              {
                  VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                           100,
                                                           ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                           constraints,
                                                           (this->fe.component_mask(this->pressure)));
              }
              else
              {
                  VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                            100,
                                                            ZeroFunction<dim>(this->n_components),
                                                            constraints,
                                                            (this->fe.component_mask(this->pressure)));
              }
              VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                        4,
                                                        ZeroFunction<dim>(this->n_components),
                                                        constraints,
                                                        this->fe.component_mask(this->z_displacement) );

            Point<dim> fix_node(0.5*this->parameters.scale, 0.5*this->parameters.scale, 0.0);
            typename DoFHandler<dim>::active_cell_iterator
            cell = this->dof_handler_ref.begin_active(), endc = this->dof_handler_ref.end();
            for (; cell != endc; ++cell)
              for (unsigned int node = 0; node < GeometryInfo<dim>::vertices_per_cell; ++node)
              {
                  if (  (abs(cell->vertex(node)[2]-fix_node[2]) < (1e-6 * this->parameters.scale))
                    &&  (abs(cell->vertex(node)[0]-fix_node[0]) < (1e-6 * this->parameters.scale)))
                      constraints.add_line(cell->vertex_dof_index(node, 0));

                  if (  (abs(cell->vertex(node)[2]-fix_node[2]) < (1e-6 * this->parameters.scale))
                    &&  (abs(cell->vertex(node)[1]-fix_node[1]) < (1e-6 * this->parameters.scale)))
                    constraints.add_line(cell->vertex_dof_index(node, 1));
              }

            if (this->parameters.load_type == "displacement")
            {
                VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                           5,
                                                           ConstantFunction<dim>(this->get_dirichlet_load(5).first,this->n_components),
                                                           constraints,
                                                           this->fe.component_mask(this->get_dirichlet_load(5).second));
            }
          }

          virtual Tensor<1,dim>
          get_neumann_traction (const types::boundary_id &boundary_id,
                                const Point<dim>         &pt,
                                const Tensor<1,dim>      &N) const
          {
              if (this->parameters.load_type == "pressure")
              {
                if (boundary_id ==  5)
                {
                    const double final_load   = this->parameters.load;
                    const double current_time = this->time.get_current();
                    const double final_time   = this->time.get_end();
                    const double num_cycles   = 3.0;

                    return  final_load/2.0 * (1.0 - std::sin(numbers::PI * (2.0*num_cycles*current_time/final_time + 0.5))) * N;
                }
              }

              (void)pt;

              return Tensor<1,dim>();
            }

          virtual types::boundary_id
          get_reaction_boundary_id_for_output() const
          {
              return 5;
          }

          virtual  std::pair<double, FEValuesExtractors::Scalar>
          get_dirichlet_load(const types::boundary_id &boundary_id) const
          {
                double displ_incr = 0;
                FEValuesExtractors::Scalar direction;

                if (boundary_id == 5)
                {
                    const double final_displ  = this->parameters.load;
                    const double current_time = this->time.get_current();
                    const double final_time   = this->time.get_end();
                    const double delta_time   = this->time.get_delta_t();
                    const double num_cycles   = 3.0;
                    double current_displ = 0.0;
                    double previous_displ = 0.0;

                    if (this->parameters.num_cycle_sets == 1)
                    {
                        current_displ  = final_displ/2.0 * (1.0 - std::sin(numbers::PI * (2.0*num_cycles*current_time/final_time + 0.5)));
                        previous_displ = final_displ/2.0 * (1.0 - std::sin(numbers::PI * (2.0*num_cycles*(current_time-delta_time)/final_time + 0.5)));
                    }
                    else
                    {
                        if ( current_time <= (final_time*1.0/3.0) )
                        {
                            current_displ  = final_displ/2.0 * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*current_time/(final_time*1.0/3.0) + 0.5)));
                            previous_displ = final_displ/2.0 * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*(current_time-delta_time)/(final_time*1.0/3.0) + 0.5)));
                        }
                        else
                        {
                            current_displ  = final_displ * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*current_time / (final_time*2.0/3.0)
                                                  - (num_cycles - 0.5) )));
                            previous_displ = final_displ * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*(current_time-delta_time) / (final_time*2.0/3.0)
                                                  - (num_cycles - 0.5))));
                        }
                    }
                    displ_incr = current_displ - previous_displ;
                    direction = this->z_displacement;
                }
                return std::make_pair(displ_incr,direction);
          }
    };

    //@sect4{Derived class: No lateral displacement in loading surfaces}
    template <int dim>
    class BrainBudday2017CubeTensionCompressionFullyFixed
          : public BrainBudday2017BaseCube<dim>
    {
        public:
          BrainBudday2017CubeTensionCompressionFullyFixed (const Parameters::AllParameters &parameters)
            : BrainBudday2017BaseCube<dim> (parameters)
          {}

          virtual ~BrainBudday2017CubeTensionCompressionFullyFixed () {}

        private:
          virtual void
          define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices)
          {
            tracked_vertices[0][0] = 0.5*this->parameters.scale;
            tracked_vertices[0][1] = 0.5*this->parameters.scale;
            tracked_vertices[0][2] = 1.0*this->parameters.scale;

            tracked_vertices[1][0] = 0.5*this->parameters.scale;
            tracked_vertices[1][1] = 0.5*this->parameters.scale;
            tracked_vertices[1][2] = 0.5*this->parameters.scale;
          }

          virtual void
          make_dirichlet_constraints(ConstraintMatrix &constraints)
          {
              if (this->time.get_timestep() < 2)
              {
                  VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                           100,
                                                           ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                           constraints,
                                                           (this->fe.component_mask(this->pressure)));
              }
              else
              {
                  VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                            100,
                                                            ZeroFunction<dim>(this->n_components),
                                                            constraints,
                                                            (this->fe.component_mask(this->pressure)));
              }

              VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                        4,
                                                        ZeroFunction<dim>(this->n_components),
                                                        constraints,
                                                      (this->fe.component_mask(this->x_displacement) |
                                                       this->fe.component_mask(this->y_displacement) |
                                                       this->fe.component_mask(this->z_displacement) ));


            if (this->parameters.load_type == "displacement")
            {
                VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                           5,
                                                           ConstantFunction<dim>(this->get_dirichlet_load(5).first,this->n_components),
                                                           constraints,
                                                           this->fe.component_mask(this->get_dirichlet_load(5).second));

               VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                          5,
                                                          ZeroFunction<dim>(this->n_components),
                                                          constraints,
                                                          (this->fe.component_mask(this->x_displacement) |
                                                           this->fe.component_mask(this->y_displacement) ));
            }
          }

          virtual Tensor<1,dim>
          get_neumann_traction (const types::boundary_id &boundary_id,
                                const Point<dim>         &pt,
                                const Tensor<1,dim>      &N) const
          {
              if (this->parameters.load_type == "pressure")
              {
                if (boundary_id ==  5)
                {
                    const double final_load   = this->parameters.load;
                    const double current_time = this->time.get_current();
                    const double final_time   = this->time.get_end();
                    const double num_cycles   = 3.0;

                    return  final_load/2.0 * (1.0 - std::sin(numbers::PI * (2.0*num_cycles*current_time/final_time + 0.5))) * N;
                }
              }

              (void)pt;

              return Tensor<1,dim>();
            }

          virtual types::boundary_id
          get_reaction_boundary_id_for_output() const
          {
              return 5;
          }

          virtual  std::pair<double, FEValuesExtractors::Scalar>
          get_dirichlet_load(const types::boundary_id &boundary_id) const
          {
                double displ_incr = 0;
                FEValuesExtractors::Scalar direction;

                if (boundary_id == 5)
                {
                    const double final_displ  = this->parameters.load;
                    const double current_time = this->time.get_current();
                    const double final_time   = this->time.get_end();
                    const double delta_time   = this->time.get_delta_t();
                    const double num_cycles   = 3.0;
                    double current_displ = 0.0;
                    double previous_displ = 0.0;

                    if (this->parameters.num_cycle_sets == 1)
                    {
                        current_displ  = final_displ/2.0 * (1.0 - std::sin(numbers::PI * (2.0*num_cycles*current_time/final_time + 0.5)));
                        previous_displ = final_displ/2.0 * (1.0 - std::sin(numbers::PI * (2.0*num_cycles*(current_time-delta_time)/final_time + 0.5)));
                    }
                    else
                    {
                        if ( current_time <= (final_time*1.0/3.0) )
                        {
                            current_displ  = final_displ/2.0 * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*current_time/(final_time*1.0/3.0) + 0.5)));
                            previous_displ = final_displ/2.0 * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*(current_time-delta_time)/(final_time*1.0/3.0) + 0.5)));
                        }
                        else
                        {
                            current_displ  = final_displ * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*current_time / (final_time*2.0/3.0)
                                                  - (num_cycles - 0.5) )));
                            previous_displ = final_displ * (1.0 - std::sin(numbers::PI *
                                                 (2.0*num_cycles*(current_time-delta_time) / (final_time*2.0/3.0)
                                                  - (num_cycles - 0.5))));
                        }
                    }
                    displ_incr = current_displ - previous_displ;
                    direction = this->z_displacement;
                }
                return std::make_pair(displ_incr,direction);
          }
    };

    //@sect4{Derived class: No lateral or vertical displacement in loading surface}
    template <int dim>
    class BrainBudday2017CubeShearFullyFixed
          : public BrainBudday2017BaseCube<dim>
    {
        public:
          BrainBudday2017CubeShearFullyFixed (const Parameters::AllParameters &parameters)
            : BrainBudday2017BaseCube<dim> (parameters)
          {}

          virtual ~BrainBudday2017CubeShearFullyFixed () {}

        private:
          virtual void
          define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices)
          {
            tracked_vertices[0][0] = 0.75*this->parameters.scale;
            tracked_vertices[0][1] = 0.5*this->parameters.scale;
            tracked_vertices[0][2] = 0.0*this->parameters.scale;

            tracked_vertices[1][0] = 0.25*this->parameters.scale;
            tracked_vertices[1][1] = 0.5*this->parameters.scale;
            tracked_vertices[1][2] = 0.0*this->parameters.scale;
          }

          virtual void
          make_dirichlet_constraints(ConstraintMatrix &constraints)
          {
              if (this->time.get_timestep() < 2)
              {
                  VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                           100,
                                                           ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                           constraints,
                                                           (this->fe.component_mask(this->pressure)));
              }
              else
              {
                  VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                            100,
                                                            ZeroFunction<dim>(this->n_components),
                                                            constraints,
                                                            (this->fe.component_mask(this->pressure)));
              }

              VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                        5,
                                                        ZeroFunction<dim>(this->n_components),
                                                        constraints,
                                                      (this->fe.component_mask(this->x_displacement) |
                                                       this->fe.component_mask(this->y_displacement) |
                                                       this->fe.component_mask(this->z_displacement) ));


            if (this->parameters.load_type == "displacement")
            {
                VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                           4,
                                                           ConstantFunction<dim>(this->get_dirichlet_load(4).first,this->n_components),
                                                           constraints,
                                                           this->fe.component_mask(this->get_dirichlet_load(4).second));

               VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                          4,
                                                          ZeroFunction<dim>(this->n_components),
                                                          constraints,
                                                          (this->fe.component_mask(this->y_displacement) |
                                                           this->fe.component_mask(this->z_displacement) ));
            }
          }

          virtual Tensor<1,dim>
          get_neumann_traction (const types::boundary_id &boundary_id,
                                const Point<dim>         &pt,
                                const Tensor<1,dim>      &N) const
          {
              if (this->parameters.load_type == "pressure")
              {
                if (boundary_id ==  4)
                {
                    const double final_load   = this->parameters.load;
                    const double current_time = this->time.get_current();
                    const double final_time   = this->time.get_end();
                    const double num_cycles   = 3.0;
                    const Point< 3, double> axis (0.0,1.0,0.0);
                    const double angle = numbers::PI;
                    static const Tensor< 2, dim, double> R(Physics::Transformations::Rotations::rotation_matrix_3d(axis,angle));

                    return  (final_load * (std::sin(2.0*(numbers::PI)*num_cycles*current_time/final_time)) * (R * N));
                }
              }

              (void)pt;

              return Tensor<1,dim>();
            }

          virtual types::boundary_id
          get_reaction_boundary_id_for_output() const
          {
              return 4;
          }

          virtual  std::pair<double, FEValuesExtractors::Scalar>
          get_dirichlet_load(const types::boundary_id &boundary_id) const
          {
                double displ_incr = 0;
                FEValuesExtractors::Scalar direction;

                if (boundary_id == 4)
                {
                    const double final_displ  = this->parameters.load;
                    const double current_time = this->time.get_current();
                    const double final_time   = this->time.get_end();
                    const double delta_time   = this->time.get_delta_t();
                    const double num_cycles   = 3.0;
                    double current_displ = 0.0;
                    double previous_displ = 0.0;

                    if (this->parameters.num_cycle_sets == 1)
                    {
                        current_displ  = final_displ * (std::sin(2.0*(numbers::PI)*num_cycles*current_time/final_time));
                        previous_displ = final_displ * (std::sin(2.0*(numbers::PI)*num_cycles*(current_time-delta_time)/final_time));
                    }
                    else
                    {
                        AssertThrow(false, ExcMessage("Problem type not defined. Budday shear experiments implemented only for one set of cycles."));
                    }
                    displ_incr = current_displ - previous_displ;
                    direction = this->x_displacement;
                }
                return std::make_pair(displ_incr,direction);
          }
    };

    // @sect3{Continuum growth examples}
    // We group the definition of the geometry, boundary and loading conditions specific to
    // the examplesrelated to continuum growth into specific classes.

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

          GridGenerator::cylinder( this->triangulation,
                                   0.2,   //radius
                                   0.2);  //half-length
          //Create a cylinder around the x-axis. The cylinder extends from x=-"half_length"
          //to x=+"half_length" and its projection into the yz-plane is a circle of radius "radius".
          //The boundaries are colored according to the following scheme:
          //0 for the hull of the cylinder, 1 for the left hand face and 2 for the right hand face.

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
         const double PI = 3.141592653589793;
         const double rot_angle = 3.0*PI/2.0;
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
         this->triangulation.refine_global(std::max (1U, this->parameters.global_refinement));
         this->triangulation.reset_manifold(0);
      }

      virtual void
      define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices)
      {
        tracked_vertices[0][0] = 0.0*this->parameters.scale;
        tracked_vertices[0][1] = 0.0*this->parameters.scale;
        tracked_vertices[0][2] = 0.25*this->parameters.scale;

        tracked_vertices[1][0] = 0.0*this->parameters.scale;
        tracked_vertices[1][1] = 0.0*this->parameters.scale;
        tracked_vertices[1][2] = 0.0*this->parameters.scale;
      }

      virtual void
      make_dirichlet_constraints(ConstraintMatrix &constraints)
      {
          if (this->time.get_timestep() < 2)
          {
            VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                     2,
                                                     ConstantFunction<dim>(this->parameters.drained_pressure,
                                                                           this->n_components),
                                                     constraints,
                                                     (this->fe.component_mask(this->pressure)));
          }
          else
          {
            VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                     2,
                                                     ZeroFunction<dim>(this->n_components),
                                                     constraints,
                                                     (this->fe.component_mask(this->pressure)));
          }

          VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                   1,
                                                   ZeroFunction<dim>(this->n_components),
                                                   constraints,
                                                   ( this->fe.component_mask(this->x_displacement) |
                                                     this->fe.component_mask(this->y_displacement) |
                                                     this->fe.component_mask(this->z_displacement) ) );

          VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                   0,
                                                   ZeroFunction<dim>(this->n_components),
                                                   constraints,
                                                   ( this->fe.component_mask(this->x_displacement) |
                                                     this->fe.component_mask(this->y_displacement) |
                                                     this->fe.component_mask(this->z_displacement) ) );

      }

      virtual Tensor<1,dim>
      get_neumann_traction (const types::boundary_id &boundary_id,
                            const Point<dim>         &pt,
                            const Tensor<1,dim>      &N) const
      {
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

      virtual types::boundary_id
      get_reaction_boundary_id_for_output() const
      {
          return 0;
      }

      virtual  std::pair<types::boundary_id,types::boundary_id>
      get_drained_boundary_id_for_output() const
      {
          return std::make_pair(2,2);
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
          GridGenerator::half_hyper_ball( this->triangulation,
                                          center,
                                          radius );
          //A half hyper-ball around center, which contains 6 in 3d.
          //The cut plane is perpendicular to the x-axis.
          //The boundary indicators are 0 for the curved boundary and 1 for the cut plane.
          //The manifold id for the curved boundary is set to zero, and a SphericalManifold is attached to it.

          //Rotate half-sphere so that it is perpendicular to the z-axis
          const double PI = 3.141592653589793;
          const double rot_angle = 3.0*PI/2.0;
          GridTools::rotate( rot_angle, 1, this->triangulation);

          //Elongate half-sphere in the x direction.
          //GridTools::transform(TransfTurtle(), this->triangulation);

         GridTools::scale(this->parameters.scale, this->triangulation);
         this->triangulation.refine_global(std::max (1U, this->parameters.global_refinement));


         //Set area for constraint
         double cirumf = PI * radius;
         double x_plane_fix = 0.1*this->parameters.scale;
         double margin = (cirumf*this->parameters.scale)/(4 * this->parameters.global_refinement);

         typename Triangulation<dim>::active_cell_iterator cell =
                 this->triangulation.begin_active(), endc = this->triangulation.end();
         for (; cell != endc; ++cell)
         {
           for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
             if ( cell->face(face)->at_boundary() == true  &&
                  cell->face(face)->center()[0] > 0        &&
                  ( ((abs(cell->face(face)->center()[0] - x_plane_fix) < 0.5*margin ) &&
                     (abs(cell->face(face)->center()[2]) < 0.5*radius*this->parameters.scale )) ||
                    ((abs(cell->face(face)->center()[0] - x_plane_fix) < 0.4*margin ) &&
                     (abs(cell->face(face)->center()[2]) > 0.5*radius*this->parameters.scale ))  ) )
                   cell->face(face)->set_boundary_id(2);
         }
      }

      virtual void
      define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices)
      {
        tracked_vertices[0][0] = 0.0*this->parameters.scale;
        tracked_vertices[0][1] = 0.0*this->parameters.scale;
        tracked_vertices[0][2] = 1.0*this->parameters.scale;

        tracked_vertices[1][0] = 0.0*this->parameters.scale;
        tracked_vertices[1][1] = 0.0*this->parameters.scale;
        tracked_vertices[1][2] = 0.0*this->parameters.scale;
      }

      virtual void
      make_dirichlet_constraints(ConstraintMatrix &constraints)
      {
          if (this->time.get_timestep() < 2)
          {
            VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                     0,
                                                     ConstantFunction<dim>(this->parameters.drained_pressure,
                                                                           this->n_components),
                                                     constraints,
                                                     (this->fe.component_mask(this->pressure)));

            VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                     1,
                                                     ConstantFunction<dim>(this->parameters.drained_pressure,
                                                                           this->n_components),
                                                     constraints,
                                                     (this->fe.component_mask(this->pressure)));
          }
          else
          {
            VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                     0,
                                                     ZeroFunction<dim>(this->n_components),
                                                     constraints,
                                                     (this->fe.component_mask(this->pressure)));

            VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                     1,
                                                     ZeroFunction<dim>(this->n_components),
                                                     constraints,
                                                     (this->fe.component_mask(this->pressure)));
          }
          VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                   1,
                                                   ZeroFunction<dim>(this->n_components),
                                                   constraints,
                                                   ( this->fe.component_mask(this->z_displacement) ) );

          VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                   2,
                                                   ZeroFunction<dim>(this->n_components),
                                                   constraints,
                                                   ( this->fe.component_mask(this->x_displacement) |
                                                     this->fe.component_mask(this->y_displacement) |
                                                     this->fe.component_mask(this->z_displacement)) );
      }

      virtual Tensor<1,dim>
      get_neumann_traction (const types::boundary_id &boundary_id,
                            const Point<dim>         &pt,
                            const Tensor<1,dim>      &N) const
      {
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

      virtual types::boundary_id
      get_reaction_boundary_id_for_output() const
      {
          return 0;
      }

      virtual  std::pair<types::boundary_id,types::boundary_id>
      get_drained_boundary_id_for_output() const
      {
          return std::make_pair(2,2);
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


    //@sect4{Brain cube}
    template <int dim>
      class GrowthBrainBaseCube
          : public Solid<dim>
    {
    public:
        GrowthBrainBaseCube (const Parameters::AllParameters &parameters)
        : Solid<dim> (parameters)
      {}

      virtual ~GrowthBrainBaseCube () {}

    private:
      virtual void
      make_grid()
      {
        GridGenerator::hyper_cube(this->triangulation,
                                  0.0,
                                  1.0,
                                  true);
        // Cube 1 x 1 x 1
        // If the colorize flag is true, the boundary_ids of the boundary faces are assigned,
        // such that the lower one in x-direction is 0, the upper one is 1. The indicators for
        // the surfaces in y-direction are 2 and 3, the ones for z are 4 and 5.

        // Assign all faces same boundary id = 0
        typename Triangulation<dim>::active_cell_iterator cell =
                this->triangulation.begin_active(), endc = this->triangulation.end();
        for (; cell != endc; ++cell)
            for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                if (cell->face(face)->at_boundary() == true)
                    cell->face(face)->set_boundary_id(0); //All surfaces have boundary id 0

        GridTools::scale(this->parameters.scale, this->triangulation);
        //this->triangulation.refine_global(std::max (1U, this->parameters.global_refinement));
        this->triangulation.refine_global(this->parameters.global_refinement);
      }

      virtual void
      define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices)
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

      virtual types::boundary_id
      get_reaction_boundary_id_for_output() const
      {
          return 0;
      }

      virtual  std::pair<types::boundary_id,types::boundary_id>
      get_drained_boundary_id_for_output() const
      {
          return std::make_pair(0,0);
      }

      virtual Tensor<1,dim>
      get_neumann_traction (const types::boundary_id &boundary_id,
                              const Point<dim>         &pt,
                              const Tensor<1,dim>      &N) const
      {
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
      class GrowthBrainConfinedDrained
          : public GrowthBrainBaseCube<dim>
    {
    public:
        GrowthBrainConfinedDrained (const Parameters::AllParameters &parameters)
        : GrowthBrainBaseCube<dim> (parameters)
      {}

      virtual ~GrowthBrainConfinedDrained () {}

    private:
      virtual void
      make_dirichlet_constraints(ConstraintMatrix &constraints)
      {
          if (this->time.get_timestep() < 2) //Dirichlet BC on pressure nodes
          {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       0,
                                                       ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
          }
          else
          {
              VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                        0,
                                                        ZeroFunction<dim>(this->n_components),
                                                        constraints,
                                                        (this->fe.component_mask(this->pressure)));
          }

          VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                    0, //bottom face: fixed z-displacements
                                                    ZeroFunction<dim>(this->n_components),
                                                    constraints,
                                                    ( this->fe.component_mask(this->x_displacement) |
                                                      this->fe.component_mask(this->y_displacement) |
                                                      this->fe.component_mask(this->z_displacement) ));
      }
    };

    template <int dim>
      class GrowthBrainConfinedUndrained
          : public GrowthBrainBaseCube<dim>
    {
    public:
        GrowthBrainConfinedUndrained (const Parameters::AllParameters &parameters)
        : GrowthBrainBaseCube<dim> (parameters)
      {}

      virtual ~GrowthBrainConfinedUndrained () {}

    private:
      virtual void
      make_dirichlet_constraints(ConstraintMatrix &constraints)
      {

          VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                    0, //bottom face: fixed z-displacements
                                                    ZeroFunction<dim>(this->n_components),
                                                    constraints,
                                                    ( this->fe.component_mask(this->x_displacement) |
                                                      this->fe.component_mask(this->y_displacement) |
                                                      this->fe.component_mask(this->z_displacement) ));
      }
    };


    template <int dim>
      class GrowthBrainUnconfinedDrained
          : public GrowthBrainBaseCube<dim>
    {
    public:
        GrowthBrainUnconfinedDrained (const Parameters::AllParameters &parameters)
        : GrowthBrainBaseCube<dim> (parameters)
      {}

      virtual ~GrowthBrainUnconfinedDrained () {}

    private:
      virtual void
      make_dirichlet_constraints(ConstraintMatrix &constraints)
      {
          if (this->time.get_timestep() < 2) //Dirichlet BC on pressure nodes
          {
              VectorTools::interpolate_boundary_values(this->dof_handler_ref,
                                                       0,
                                                       ConstantFunction<dim>(this->parameters.drained_pressure,this->n_components),
                                                       constraints,
                                                       (this->fe.component_mask(this->pressure)));
          }
          else
          {
              VectorTools::interpolate_boundary_values( this->dof_handler_ref,
                                                        0,
                                                        ZeroFunction<dim>(this->n_components),
                                                        constraints,
                                                        (this->fe.component_mask(this->pressure)));
          }

          // Fully-fix a node at the center of the cube
          Point<dim> fix_node(0.5*this->parameters.scale, 0.5*this->parameters.scale, 0.5*this->parameters.scale);
          typename DoFHandler<dim>::active_cell_iterator
          cell = this->dof_handler_ref.begin_active(), endc = this->dof_handler_ref.end();
          for (; cell != endc; ++cell)
            for (unsigned int node = 0; node < GeometryInfo<dim>::vertices_per_cell; ++node)
            {
                if (  (abs(cell->vertex(node)[0]-fix_node[0]) < (1e-6 * this->parameters.scale))
                  &&  (abs(cell->vertex(node)[1]-fix_node[1]) < (1e-6 * this->parameters.scale))
                  &&  (abs(cell->vertex(node)[2]-fix_node[2]) < (1e-6 * this->parameters.scale)) )
                {
                    constraints.add_line(cell->vertex_dof_index(node, 0));
                    constraints.add_line(cell->vertex_dof_index(node, 1));
                    constraints.add_line(cell->vertex_dof_index(node, 2));

                }
            }
      }
    };

    template <int dim>
      class GrowthBrainUnconfinedUndrained
          : public GrowthBrainBaseCube<dim>
    {
    public:
        GrowthBrainUnconfinedUndrained (const Parameters::AllParameters &parameters)
        : GrowthBrainBaseCube<dim> (parameters)
      {}

      virtual ~GrowthBrainUnconfinedUndrained () {}

    private:
      virtual void
      make_dirichlet_constraints(ConstraintMatrix &constraints)
      {
          // Fully-fix a node at the center of the cube
          Point<dim> fix_node(0.5*this->parameters.scale, 0.5*this->parameters.scale, 0.5*this->parameters.scale);
          typename DoFHandler<dim>::active_cell_iterator
          cell = this->dof_handler_ref.begin_active(), endc = this->dof_handler_ref.end();
          for (; cell != endc; ++cell)
            for (unsigned int node = 0; node < GeometryInfo<dim>::vertices_per_cell; ++node)
            {
                if (  (abs(cell->vertex(node)[0]-fix_node[0]) < (1e-6 * this->parameters.scale))
                  &&  (abs(cell->vertex(node)[1]-fix_node[1]) < (1e-6 * this->parameters.scale))
                  &&  (abs(cell->vertex(node)[2]-fix_node[2]) < (1e-6 * this->parameters.scale)) )
                {
                    constraints.add_line(cell->vertex_dof_index(node, 0));
                    constraints.add_line(cell->vertex_dof_index(node, 1));
                    constraints.add_line(cell->vertex_dof_index(node, 2));

                }
            }
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
      if (parameters.geom_type == "Ehlers_tube_step_load")
      {
        ValidationEhlers1999StepLoad<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "Ehlers_tube_increase_load")
      {
        ValidationEhlers1999IncreaseLoad<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "Ehlers_cube_consolidation")
      {
        ValidationEhlers1999CubeConsolidation<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "Franceschini_consolidation")
      {
        Franceschini2006Consolidation<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "Budday_cube_tension_compression")
      {
        BrainBudday2017CubeTensionCompression<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "Budday_cube_tension_compression_fully_fixed")
      {
        BrainBudday2017CubeTensionCompressionFullyFixed<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "Budday_cube_shear_fully_fixed")
      {
        BrainBudday2017CubeShearFullyFixed<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "growing_muffin")
      {
        GrowingMuffin<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "trapped_turtle")
      {
        TrappedTurtle<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "brain_growth_confined_drained")
      {
        GrowthBrainConfinedDrained<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "brain_growth_confined_undrained")
      {
        GrowthBrainConfinedUndrained<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "brain_growth_unconfined_drained")
      {
        GrowthBrainUnconfinedDrained<3> solid_3d(parameters);
        solid_3d.run();
      }
      else if (parameters.geom_type == "brain_growth_unconfined_undrained")
      {
        GrowthBrainUnconfinedUndrained<3> solid_3d(parameters);
        solid_3d.run();
      }
      else
      {
        AssertThrow(false, ExcMessage("Problem type not defined. Current setting: " + parameters.geom_type));
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
