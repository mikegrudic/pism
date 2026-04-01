#include "../allvars.h"
#include "../proto.h"
#include "indices.h"
#include "math.h"
#include "microphysics_func_jac.h"
#include "string.h"
#include <string.h>

#ifdef JACO

void call_jaco(int i) {
    jaco_do_cooling(i);
    SphP[i].InternalEnergyPred = SphP[i].InternalEnergy;
    jaco_set_eos_pressure(i);
#ifndef COOLING_OPERATOR_SPLIT
    if (SphP[i].CoolingIsOperatorSplitThisTimestep == 0) {
        SphP[i].DtInternalEnergy = 0;
    } // if unsplit, zero the internal energy change here
#endif
}

void jaco_do_cooling(int i) {
    double dtime = GET_PARTICLE_TIMESTEP_IN_PHYSICAL(i);
    if (dtime == 0) {
        return;
    }
    double Δt = dtime * UNIT_TIME_IN_CGS;
    set_PdV_work_heatingrate(i, dtime);
    double u_0 = SphP[i].InternalEnergy * UNIT_SPECEGY_IN_CGS, T = SphP[i].InternalEnergy * U_TO_TEMP_UNITS, u = u_0,
           n_Htot = nH_CGS(i), x_H = 1., pdv_work = 0;
    if (SphP[i].CoolingIsOperatorSplitThisTimestep == 0) {
        pdv_work = SphP[i].DtInternalEnergy * n_Htot;
    }

#include "assignments.h" // assign initial values of X and params
    int num_iter = jaco_solve(X, params, 1e-6);
    SphP[i].InternalEnergy = X[INDEX_u] / UNIT_SPECEGY_IN_CGS;
    double temp = X[INDEX_T];
}

void jaco_set_eos_pressure(int i) {
    SphP[i].Pressure = (GAMMA(i) - 1) * SphP[i].InternalEnergyPred * Get_Gas_density_for_energy_i(i);
}

int iter_condition(double *X, double *dx, double tol) {
    int cond = 0;
    for (int i = 0; i < NUM_VARS; i++) {
        cond += (fabs(dx[i]) > tol * fabs(X[i]));
    }
    return cond;
}

int solve_failure_condition(double *X, double *dx, double tol, int num_iter) {
    int failure = 0;
    failure |= num_iter >= MAXITER;
    failure |= X[INDEX_T] > 1e11;
    failure |= X[INDEX_u] > 1e18;
    for (int i; i < 0; i++) {
        failure |= isnan(dx[i]);
        failure |= isnan(X[i]);
    }
    return failure;
}

int jaco_solve(double *X, double *params, double tol) {
    double funcjac[NUM_VARS * (NUM_VARS + 1)], func[NUM_VARS], jac[NUM_VARS * NUM_VARS], jacinv[NUM_VARS * NUM_VARS],
        dx[NUM_VARS] = {MAX_REAL_NUMBER, MAX_REAL_NUMBER}, X0[NUM_VARS], T_upper = MAX_REAL_NUMBER,
        T_lower = MIN_REAL_NUMBER;

    memcpy(X0, X, NUM_VARS * sizeof(double));
    int num_iter = 0;
    int careful_steps = 1;
    while (iter_condition(X, dx, tol)) {
        microphysics_func_jac(X, params, funcjac);
        memcpy(func, funcjac, NUM_VARS * sizeof(double));
        memcpy(jac, funcjac + NUM_VARS, NUM_VARS * NUM_VARS * sizeof(double));
        if (func[INDEX_T] < 0) {
            T_upper = fmin(T_upper, X[INDEX_T]);
        } else { // TODO: check that func[INDEX_T] is always the heat equation!
            T_lower = fmax(T_lower, X[INDEX_T]);
        }
        double det = jac[0] * jac[3] - jac[1] * jac[2];
        jacinv[0] = jac[3] / det;
        jacinv[3] = jac[0] / det;
        jacinv[1] = -jac[1] / det;
        jacinv[2] = -jac[2] / det;
        dx[0] = -(jacinv[0] * func[0] + jacinv[1] * func[1]);
        dx[1] = -(jacinv[2] * func[0] + jacinv[3] * func[1]);
        if ((X[INDEX_T] == All.MinGasTemp) && (dx[INDEX_T] < 0) && (dx[INDEX_u] < 0)) {
            break;
        }
        const double fac = fmin(1, ((float)num_iter + 1) / careful_steps);
        X[INDEX_T] = fmax(All.MinGasTemp, fac * dx[INDEX_T] + X[INDEX_T]);
        X[INDEX_u] = fmax(All.MinEgySpec, fac * dx[INDEX_u] + X[INDEX_u]);
        num_iter++;
        if (solve_failure_condition(X, dx, tol, num_iter)) {
            if (careful_steps == 1) { // if we failed after trying an aggressive solve, start over more carefully
                careful_steps = 30;
                for (int i = 0; i < NUM_VARS; i++) {
                    X[i] = X0[i];
                    dx[i] = MAX_REAL_NUMBER;
                }
                num_iter = 0;
                continue;                     // try again
            } else if (careful_steps == 30) { // perhaps we're getting stuck in the trough of the cooling curve
                X[INDEX_T] = sqrt(T_upper * T_lower);
                X[INDEX_u] = X[INDEX_T] / U_TO_TEMP_UNITS;
                dx[0] = dx[1] = MAX_REAL_NUMBER;
                num_iter = 0;
                continue; // try again
            } else {
                printf("jaco failed to converge for num_iter=%d n=%g u0=%g u=%g T=%g X0=%g %g X=%g %g dx=%g %g func=%g "
                       "%g careful_steps=%d",
                       num_iter, params[INDEX_n_Htot], params[INDEX_u_0], X[INDEX_u], X[INDEX_T], X0[0], X0[1], X[0],
                       X[1], dx[0], dx[1], func[0], func[1], careful_steps);
                endrun(10);
            }
        }
    }
    return num_iter;
}
#endif