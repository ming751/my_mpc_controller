/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific
#include "robot_arm_model/robot_arm_model.h"


#include "robot_arm_cost/robot_arm_cost.h"



#include "acados_solver_robot_arm.h"

#define NX     ROBOT_ARM_NX
#define NZ     ROBOT_ARM_NZ
#define NU     ROBOT_ARM_NU
#define NP     ROBOT_ARM_NP
#define NP_GLOBAL     ROBOT_ARM_NP_GLOBAL
#define NY0    ROBOT_ARM_NY0
#define NY     ROBOT_ARM_NY
#define NYN    ROBOT_ARM_NYN

#define NBX    ROBOT_ARM_NBX
#define NBX0   ROBOT_ARM_NBX0
#define NBU    ROBOT_ARM_NBU
#define NG     ROBOT_ARM_NG
#define NBXN   ROBOT_ARM_NBXN
#define NGN    ROBOT_ARM_NGN

#define NH     ROBOT_ARM_NH
#define NHN    ROBOT_ARM_NHN
#define NH0    ROBOT_ARM_NH0
#define NPHI   ROBOT_ARM_NPHI
#define NPHIN  ROBOT_ARM_NPHIN
#define NPHI0  ROBOT_ARM_NPHI0
#define NR     ROBOT_ARM_NR

#define NS     ROBOT_ARM_NS
#define NS0    ROBOT_ARM_NS0
#define NSN    ROBOT_ARM_NSN

#define NSBX   ROBOT_ARM_NSBX
#define NSBU   ROBOT_ARM_NSBU
#define NSH0   ROBOT_ARM_NSH0
#define NSH    ROBOT_ARM_NSH
#define NSHN   ROBOT_ARM_NSHN
#define NSG    ROBOT_ARM_NSG
#define NSPHI0 ROBOT_ARM_NSPHI0
#define NSPHI  ROBOT_ARM_NSPHI
#define NSPHIN ROBOT_ARM_NSPHIN
#define NSGN   ROBOT_ARM_NSGN
#define NSBXN  ROBOT_ARM_NSBXN



// ** solver data **

robot_arm_solver_capsule * robot_arm_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(robot_arm_solver_capsule));
    robot_arm_solver_capsule *capsule = (robot_arm_solver_capsule *) capsule_mem;

    return capsule;
}


int robot_arm_acados_free_capsule(robot_arm_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int robot_arm_acados_create(robot_arm_solver_capsule* capsule)
{
    int N_shooting_intervals = ROBOT_ARM_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return robot_arm_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int robot_arm_acados_update_time_steps(robot_arm_solver_capsule* capsule, int N, double* new_time_steps)
{
    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "robot_arm_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;
}

/**
 * Internal function for robot_arm_acados_create: step 1
 */
void robot_arm_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;

    nlp_solver_plan->nlp_cost[0] = NONLINEAR_LS;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = NONLINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = NONLINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = DISCRETE_MODEL;
        // discrete dynamics does not need sim solver option, this field is ignored
        nlp_solver_plan->sim_solver_plan[i].sim_solver = INVALID_SIM_SOLVER;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;

    nlp_solver_plan->globalization = FIXED_STEP;
}


static ocp_nlp_dims* robot_arm_acados_create_setup_dimensions(robot_arm_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    nbxe[0] = 0;
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);

    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for robot_arm_acados_create: step 3
 */
void robot_arm_acados_create_setup_functions(robot_arm_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    ext_fun_opts.external_workspace = true;
    // nonlinear least squares function
    MAP_CASADI_FNC(cost_y_0_fun, robot_arm_cost_y_0_fun);
    MAP_CASADI_FNC(cost_y_0_fun_jac_ut_xt, robot_arm_cost_y_0_fun_jac_ut_xt);




    // discrete dynamics
    capsule->discr_dyn_phi_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_fun[i], robot_arm_dyn_disc_phi_fun);
    }

    capsule->discr_dyn_phi_fun_jac_ut_xt = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt[i], robot_arm_dyn_disc_phi_fun_jac);
    }

  

  
    // nonlinear least squares cost
    capsule->cost_y_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++)
    {
        MAP_CASADI_FNC(cost_y_fun[i], robot_arm_cost_y_fun);
    }

    capsule->cost_y_fun_jac_ut_xt = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++)
    {
        MAP_CASADI_FNC(cost_y_fun_jac_ut_xt[i], robot_arm_cost_y_fun_jac_ut_xt);
    }
    // nonlinear least square function
    MAP_CASADI_FNC(cost_y_e_fun, robot_arm_cost_y_e_fun);
    MAP_CASADI_FNC(cost_y_e_fun_jac_ut_xt, robot_arm_cost_y_e_fun_jac_ut_xt);

#undef MAP_CASADI_FNC
}


/**
 * Internal function for robot_arm_acados_create: step 4
 */
void robot_arm_acados_create_set_default_parameters(robot_arm_solver_capsule* capsule)
{

    const int N = capsule->nlp_solver_plan->N;
    // initialize parameters to nominal value
    double* p = calloc(NP, sizeof(double));

    for (int i = 0; i <= N; i++) {
        robot_arm_acados_update_params(capsule, i, p, NP);
    }
    free(p);


    // no global parameters defined
}


/**
 * Internal function for robot_arm_acados_create: step 5
 */
void robot_arm_acados_setup_nlp_in(robot_arm_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
//    ocp_nlp_in * nlp_in = ocp_nlp_in_create(nlp_config, nlp_dims);
//    capsule->nlp_in = nlp_in;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        robot_arm_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps
    double time_step = 0.05;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
        }
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 0.05;
        cost_scaling[1] = 0.05;
        cost_scaling[2] = 0.05;
        cost_scaling[3] = 0.05;
        cost_scaling[4] = 0.05;
        cost_scaling[5] = 0.05;
        cost_scaling[6] = 0.05;
        cost_scaling[7] = 0.05;
        cost_scaling[8] = 0.05;
        cost_scaling[9] = 0.05;
        cost_scaling[10] = 0.05;
        cost_scaling[11] = 0.05;
        cost_scaling[12] = 0.05;
        cost_scaling[13] = 0.05;
        cost_scaling[14] = 0.05;
        cost_scaling[15] = 0.05;
        cost_scaling[16] = 0.05;
        cost_scaling[17] = 0.05;
        cost_scaling[18] = 0.05;
        cost_scaling[19] = 0.05;
        cost_scaling[20] = 0.05;
        cost_scaling[21] = 0.05;
        cost_scaling[22] = 0.05;
        cost_scaling[23] = 0.05;
        cost_scaling[24] = 0.05;
        cost_scaling[25] = 0.05;
        cost_scaling[26] = 0.05;
        cost_scaling[27] = 0.05;
        cost_scaling[28] = 0.05;
        cost_scaling[29] = 0.05;
        cost_scaling[30] = 0.05;
        cost_scaling[31] = 0.05;
        cost_scaling[32] = 0.05;
        cost_scaling[33] = 0.05;
        cost_scaling[34] = 0.05;
        cost_scaling[35] = 0.05;
        cost_scaling[36] = 0.05;
        cost_scaling[37] = 0.05;
        cost_scaling[38] = 0.05;
        cost_scaling[39] = 0.05;
        cost_scaling[40] = 0.05;
        cost_scaling[41] = 0.05;
        cost_scaling[42] = 0.05;
        cost_scaling[43] = 0.05;
        cost_scaling[44] = 0.05;
        cost_scaling[45] = 0.05;
        cost_scaling[46] = 0.05;
        cost_scaling[47] = 0.05;
        cost_scaling[48] = 0.05;
        cost_scaling[49] = 0.05;
        cost_scaling[50] = 0.05;
        cost_scaling[51] = 0.05;
        cost_scaling[52] = 0.05;
        cost_scaling[53] = 0.05;
        cost_scaling[54] = 0.05;
        cost_scaling[55] = 0.05;
        cost_scaling[56] = 0.05;
        cost_scaling[57] = 0.05;
        cost_scaling[58] = 0.05;
        cost_scaling[59] = 0.05;
        cost_scaling[60] = 0.05;
        cost_scaling[61] = 0.05;
        cost_scaling[62] = 0.05;
        cost_scaling[63] = 0.05;
        cost_scaling[64] = 0.05;
        cost_scaling[65] = 0.05;
        cost_scaling[66] = 0.05;
        cost_scaling[67] = 0.05;
        cost_scaling[68] = 0.05;
        cost_scaling[69] = 0.05;
        cost_scaling[70] = 0.05;
        cost_scaling[71] = 0.05;
        cost_scaling[72] = 0.05;
        cost_scaling[73] = 0.05;
        cost_scaling[74] = 0.05;
        cost_scaling[75] = 0.05;
        cost_scaling[76] = 0.05;
        cost_scaling[77] = 0.05;
        cost_scaling[78] = 0.05;
        cost_scaling[79] = 0.05;
        cost_scaling[80] = 0.05;
        cost_scaling[81] = 0.05;
        cost_scaling[82] = 0.05;
        cost_scaling[83] = 0.05;
        cost_scaling[84] = 0.05;
        cost_scaling[85] = 0.05;
        cost_scaling[86] = 0.05;
        cost_scaling[87] = 0.05;
        cost_scaling[88] = 0.05;
        cost_scaling[89] = 0.05;
        cost_scaling[90] = 0.05;
        cost_scaling[91] = 0.05;
        cost_scaling[92] = 0.05;
        cost_scaling[93] = 0.05;
        cost_scaling[94] = 0.05;
        cost_scaling[95] = 0.05;
        cost_scaling[96] = 0.05;
        cost_scaling[97] = 0.05;
        cost_scaling[98] = 0.05;
        cost_scaling[99] = 0.05;
        cost_scaling[100] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }


    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun", &capsule->discr_dyn_phi_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun_jac",
                                   &capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
        
        
    }

    /**** Cost ****/
    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);

   double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(NY0) * 0] = 5000;
    W_0[1+(NY0) * 1] = 5000;
    W_0[2+(NY0) * 2] = 5000;
    W_0[3+(NY0) * 3] = 0.01;
    W_0[4+(NY0) * 4] = 0.01;
    W_0[5+(NY0) * 5] = 0.01;
    W_0[6+(NY0) * 6] = 0.01;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);
    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    W[0+(NY) * 0] = 5000;
    W[1+(NY) * 1] = 5000;
    W[2+(NY) * 2] = 5000;
    W[3+(NY) * 3] = 0.01;
    W[4+(NY) * 4] = 0.01;
    W[5+(NY) * 5] = 0.01;
    W[6+(NY) * 6] = 0.01;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
    double* yref_e = calloc(NYN, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);

    double* W_e = calloc(NYN*NYN, sizeof(double));
    // change only the non-zero elements:
    W_e[0+(NYN) * 0] = 50000;
    W_e[1+(NYN) * 1] = 50000;
    W_e[2+(NYN) * 2] = 50000;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun", &capsule->cost_y_0_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun_jac", &capsule->cost_y_0_fun_jac_ut_xt);
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun", &capsule->cost_y_fun[i-1]);
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun_jac", &capsule->cost_y_fun_jac_ut_xt[i-1]);
    }
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun", &capsule->cost_y_e_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun_jac", &capsule->cost_y_e_fun_jac_ut_xt);






    /**** Constraints ****/

    // bounds for initial stage








    /* constraints that are the same for initial and intermediate */
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    idxbu[0] = 0;
    idxbu[1] = 1;
    idxbu[2] = 2;
    idxbu[3] = 3;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    lbu[0] = -20;
    ubu[0] = 20;
    lbu[1] = -20;
    ubu[1] = 20;
    lbu[2] = -20;
    ubu[2] = 20;
    lbu[3] = -20;
    ubu[3] = 20;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);








    // x
    int* idxbx = malloc(NBX * sizeof(int));
    idxbx[0] = 0;
    idxbx[1] = 1;
    idxbx[2] = 2;
    idxbx[3] = 3;
    idxbx[4] = 4;
    idxbx[5] = 5;
    idxbx[6] = 6;
    idxbx[7] = 7;
    double* lubx = calloc(2*NBX, sizeof(double));
    double* lbx = lubx;
    double* ubx = lubx + NBX;
    lbx[0] = -1.2;
    ubx[0] = 2.3;
    lbx[1] = -0.15;
    ubx[1] = 1;
    ubx[2] = 1;
    ubx[3] = 2;
    lbx[4] = -10;
    ubx[4] = 10;
    lbx[5] = -10;
    ubx[5] = 10;
    lbx[6] = -10;
    ubx[6] = 10;
    lbx[7] = -10;
    ubx[7] = 10;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbx", idxbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbx", lbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubx", ubx);
    }
    free(idxbx);
    free(lubx);







    /* terminal constraints */













}


static void robot_arm_acados_create_set_opts(robot_arm_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/



    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);

    double globalization_fixed_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_fixed_step_length", &globalization_fixed_step_length);




    int with_solution_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params", &with_solution_sens_wrt_params);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 100;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    int log_primal_step_norm = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "log_primal_step_norm", &log_primal_step_norm);

    double nlp_solver_tol_min_step_norm = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_min_step_norm", &nlp_solver_tol_min_step_norm);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");




    // set SQP specific options
    double nlp_solver_tol_stat = 0.0001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_stat", &nlp_solver_tol_stat);

    double nlp_solver_tol_eq = 0.0001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_eq", &nlp_solver_tol_eq);

    double nlp_solver_tol_ineq = 0.0001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_ineq", &nlp_solver_tol_ineq);

    double nlp_solver_tol_comp = 0.0001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_comp", &nlp_solver_tol_comp);

    int nlp_solver_max_iter = 100;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "max_iter", &nlp_solver_max_iter);

    // set options for adaptive Levenberg-Marquardt Update
    bool with_adaptive_levenberg_marquardt = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_adaptive_levenberg_marquardt", &with_adaptive_levenberg_marquardt);

    double adaptive_levenberg_marquardt_lam = 5;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_lam", &adaptive_levenberg_marquardt_lam);

    double adaptive_levenberg_marquardt_mu_min = 0.0000000000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_mu_min", &adaptive_levenberg_marquardt_mu_min);

    double adaptive_levenberg_marquardt_mu0 = 0.001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_mu0", &adaptive_levenberg_marquardt_mu0);

    bool eval_residual_at_max_iter = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "eval_residual_at_max_iter", &eval_residual_at_max_iter);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);



    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
}


/**
 * Internal function for robot_arm_acados_create: step 7
 */
void robot_arm_acados_set_nlp_out(robot_arm_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with zeros

    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for robot_arm_acados_create: step 9
 */
int robot_arm_acados_create_precompute(robot_arm_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int robot_arm_acados_create_with_discretization(robot_arm_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != ROBOT_ARM_N && !new_time_steps) {
        fprintf(stderr, "robot_arm_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, ROBOT_ARM_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    robot_arm_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = robot_arm_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    robot_arm_acados_create_set_opts(capsule);

    // 4) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 5) setup functions, nlp_in and default parameters
    robot_arm_acados_create_setup_functions(capsule);
    robot_arm_acados_setup_nlp_in(capsule, N, new_time_steps);
    robot_arm_acados_create_set_default_parameters(capsule);

    // 6) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // 7) create and set nlp_out
    // 7.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 7.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    robot_arm_acados_set_nlp_out(capsule);

    // 8) do precomputations
    int status = robot_arm_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int robot_arm_acados_update_qp_solver_cond_N(robot_arm_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from robot_arm_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = robot_arm_acados_create_precompute(capsule);
    return status;
}


int robot_arm_acados_reset(robot_arm_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros

    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+2*NS0+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NH0+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "pi", buffer);
        }
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    free(buffer);
    return 0;
}




int robot_arm_acados_update_params(robot_arm_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 3;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int robot_arm_acados_update_params_sparse(robot_arm_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int robot_arm_acados_set_p_global_and_precompute_dependencies(robot_arm_solver_capsule* capsule, double* data, int data_len)
{

    printf("p_global is not defined, robot_arm_acados_set_p_global_and_precompute_dependencies does nothing.\n");
}




int robot_arm_acados_solve(robot_arm_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}


void robot_arm_acados_batch_solve(robot_arm_solver_capsule ** capsules, int * status_out, int N_batch)
{

    for (int i = 0; i < N_batch; i++)
    {
        status_out[i] = ocp_nlp_solve(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }


    return;
}


void robot_arm_acados_batch_eval_params_jac(robot_arm_solver_capsule ** capsules, int N_batch)
{

    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_eval_params_jac(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }


    return;
}



void robot_arm_acados_batch_eval_solution_sens_adj_p(robot_arm_solver_capsule ** capsules, const char *field, int stage, double *out, int offset, int N_batch)
{


    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_eval_solution_sens_adj_p(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->sens_out, field, stage, out + i*offset);
    }


    return;
}


void robot_arm_acados_batch_set_flat(robot_arm_solver_capsule ** capsules, const char *field, double *data, int N_data, int N_batch)
{
    int offset = ocp_nlp_dims_get_total_from_attr(capsules[0]->nlp_solver->config, capsules[0]->nlp_solver->dims, capsules[0]->nlp_out, field);

    if (N_batch*offset != N_data)
    {
        printf("batch_set_flat: wrong input dimension, expected %d, got %d\n", N_batch*offset, N_data);
        exit(1);
    }


    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_set_all(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out, field, data + i * offset);
    }


    return;
}



void robot_arm_acados_batch_get_flat(robot_arm_solver_capsule ** capsules, const char *field, double *data, int N_data, int N_batch)
{
    int offset = ocp_nlp_dims_get_total_from_attr(capsules[0]->nlp_solver->config, capsules[0]->nlp_solver->dims, capsules[0]->nlp_out, field);

    if (N_batch*offset != N_data)
    {
        printf("batch_get_flat: wrong input dimension, expected %d, got %d\n", N_batch*offset, N_data);
        exit(1);
    }


    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_get_all(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out, field, data + i * offset);
    }


    return;
}


int robot_arm_acados_free(robot_arm_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun[i]);
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
        
        
    }
    free(capsule->discr_dyn_phi_fun);
    free(capsule->discr_dyn_phi_fun_jac_ut_xt);
  
  

    // cost
    external_function_external_param_casadi_free(&capsule->cost_y_0_fun);
    external_function_external_param_casadi_free(&capsule->cost_y_0_fun_jac_ut_xt);
    for (int i = 0; i < N - 1; i++)
    {
        external_function_external_param_casadi_free(&capsule->cost_y_fun[i]);
        external_function_external_param_casadi_free(&capsule->cost_y_fun_jac_ut_xt[i]);
    }
    free(capsule->cost_y_fun);
    free(capsule->cost_y_fun_jac_ut_xt);
    external_function_external_param_casadi_free(&capsule->cost_y_e_fun);
    external_function_external_param_casadi_free(&capsule->cost_y_e_fun_jac_ut_xt);

    // constraints



    return 0;
}


void robot_arm_acados_print_stats(robot_arm_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    double stat[1200];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\talpha");
    if (stat_n > 8)
        printf("\t\tqp_res_stat\tqp_res_eq\tqp_res_ineq\tqp_res_comp");
    printf("\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            if (j == 0 || j == 5 || j == 6)
            {
                tmp_int = (int) stat[i + j * nrow];
                printf("%d\t", tmp_int);
            }
            else
            {
                printf("%e\t", stat[i + j * nrow]);
            }
        }
        printf("\n");
    }
}

int robot_arm_acados_custom_update(robot_arm_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *robot_arm_acados_get_nlp_in(robot_arm_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *robot_arm_acados_get_nlp_out(robot_arm_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *robot_arm_acados_get_sens_out(robot_arm_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *robot_arm_acados_get_nlp_solver(robot_arm_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *robot_arm_acados_get_nlp_config(robot_arm_solver_capsule* capsule) { return capsule->nlp_config; }
void *robot_arm_acados_get_nlp_opts(robot_arm_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *robot_arm_acados_get_nlp_dims(robot_arm_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *robot_arm_acados_get_nlp_plan(robot_arm_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
