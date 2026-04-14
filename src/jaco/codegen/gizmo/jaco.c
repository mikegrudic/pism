/*
 * jaco.c — Implicit backward-Euler microphysics solver for GIZMO.
 *
 * This file is a static template distributed with the jaco codegen package.
 * It provides the Newton-Raphson solver loop and the GIZMO integration layer.
 * The physics (RHS + Jacobian) is supplied by the codegen-produced
 * microphysics_func_jac(), which is called as a black box.
 *
 * The solver uses SolveVars and Params unions (defined in microphysics_func_jac.h)
 * that provide both named field access (sv->T, pr->n_Htot) and indexed array
 * access (sv->data[i]) via anonymous structs. This eliminates index-mismatch
 * bugs while allowing generic loops over variables.
 *
 * Recovery strategy on convergence failure:
 *   1. Full Newton steps (careful_steps=1). Fast when it works.
 *   2. Damped Newton with 30-step ramp (careful_steps=30). Prevents overshoot.
 *   3. Bisection restart (careful_steps=31). Resets to geometric mean of
 *      temperature bounds, then damped Newton again.
 *   4. If all attempts exhaust MAXITER, accept the state if the residual is
 *      small (relaxed tolerance), otherwise abort.
 */

#include "../allvars.h"
#include "../proto.h"
#include "microphysics_func_jac.h"
#include <math.h>
#include <string.h>

#ifdef JACO

/* ---- GIZMO interface layer ---- */

void gizmo_to_jaco(int i, SolveVars *sv, Params *pr) {
    /* Pack GIZMO particle data into solver structs.
       All solver quantities are in CGS; the conversion from code units happens here. */
    double dtime = GET_PARTICLE_TIMESTEP_IN_PHYSICAL(i);
    double Δt = dtime * UNIT_TIME_IN_CGS;
    set_PdV_work_heatingrate(i, dtime);
    double n_Htot = nH_CGS(i);
    sv->u = SphP[i].InternalEnergy * UNIT_SPECEGY_IN_CGS;
    sv->T = SphP[i].InternalEnergy * U_TO_TEMP_UNITS;
    pr->u_0 = sv->u;
    pr->n_Htot = n_Htot;
    pr->x_H = 1.;
    pr->Δt = Δt;
    pr->pdv_work = (SphP[i].CoolingIsOperatorSplitThisTimestep == 0) ? SphP[i].DtInternalEnergy * n_Htot : 0;
}

void jaco_to_gizmo(int i, const SolveVars *sv, const Params *pr) {
    /* Unpack solver result back into GIZMO particle data.
       Updates internal energy, predicted energy, pressure, and zeros
       DtInternalEnergy if the cooling is not operator-split.
       jaco_eos_pressure is codegen'd from the model's species composition. */
    SphP[i].InternalEnergy = sv->u / UNIT_SPECEGY_IN_CGS;
    SphP[i].InternalEnergyPred = SphP[i].InternalEnergy;
    SphP[i].Pressure = jaco_eos_pressure(sv, pr) / UNIT_PRESSURE_IN_CGS;
#ifndef COOLING_OPERATOR_SPLIT
    if (SphP[i].CoolingIsOperatorSplitThisTimestep == 0) {
        SphP[i].DtInternalEnergy = 0;
    }
#endif
}

void call_jaco(int i) {
    /* Top-level entry point called by GIZMO's cooling routine for each particle. */
    if (GET_PARTICLE_TIMESTEP_IN_PHYSICAL(i) == 0) {
        return;
    }
    SolveVars sv;
    Params pr;
    gizmo_to_jaco(i, &sv, &pr);
    jaco_solve(&sv, &pr, 1e-6);
    jaco_to_gizmo(i, &sv, &pr);
}

/* ---- Newton-Raphson solver ---- */

int iter_condition(const SolveVars *sv, const SolveVars *dsv, double tol) {
    /* Returns nonzero if any solved variable changed by more than tol * |value|
       on the last iteration (i.e. not yet converged). */
    for (int i = 0; i < N_VARS; i++) {
        if (fabs(dsv->data[i]) > tol * fabs(sv->data[i])) return 1;
    }
    return 0;
}

int solve_failure_condition(const SolveVars *sv, const SolveVars *dsv, double tol, int num_iter) {
    /* Returns nonzero if the solver should give up on the current attempt:
       exceeded MAXITER, variable out of physical bounds, or NaN detected. */
    int failure = 0;
    failure |= num_iter >= MAXITER;
    failure |= sv->T > 1e11;
    failure |= sv->u > 1e18;
    for (int i = 0; i < N_VARS; i++) {
        failure |= isnan(dsv->data[i]);
        failure |= isnan(sv->data[i]);
    }
    return failure;
}

int jaco_solve(SolveVars *sv, const Params *pr, double tol) {
    /* Newton-Raphson solver for the backward-Euler microphysics step.
     *
     * Solves F(X) = 0 where F is the residual from microphysics_func_jac():
     *   F_T: energy equation (heating - cooling - implicit time derivative)
     *   F_u: equation of state (u = k_B*T / (mu*m_H))
     *
     * The Newton step is dx = -J^{-1} F, applied with optional damping (fac < 1)
     * and floor clamping (MinGasTemp, MinEgySpec).
     *
     * T_upper/T_lower track bounds on the root from the sign of F_T, used for
     * bisection recovery when Newton oscillates.
     */
    SolveVars func, dsv;
    const SolveVars sv0 = *sv;
    for (int i = 0; i < N_VARS; i++) dsv.data[i] = MAX_REAL_NUMBER;
    double jac[N_VARS][N_VARS], jacinv[N_VARS * N_VARS];
    double T_upper = MAX_REAL_NUMBER, T_lower = MIN_REAL_NUMBER;

    int num_iter = 0;
    int careful_steps = 1;
    while (iter_condition(sv, &dsv, tol)) {
        microphysics_func_jac(sv, pr, &func, jac);

        /* Track temperature bounds from sign of energy equation residual */
        if (func.T < 0) {
            T_upper = fmin(T_upper, sv->T);
        } else {
            T_lower = fmax(T_lower, sv->T);
        }

        /* 2x2 matrix inverse — hardcoded for the current 2-variable system.
           TODO: generalize for N_VARS > 2 in the codegen. */
        double det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
        jacinv[0] = jac[1][1] / det;
        jacinv[3] = jac[0][0] / det;
        jacinv[1] = -jac[0][1] / det;
        jacinv[2] = -jac[1][0] / det;
        for (int i = 0; i < N_VARS; i++) {
            dsv.data[i] = 0;
            for (int j = 0; j < N_VARS; j++) {
                dsv.data[i] -= jacinv[i * N_VARS + j] * func.data[j];
            }
        }

        /* At the temperature floor with a negative step, we can't go lower — done */
        if ((sv->T <= All.MinGasTemp) && (dsv.T < 0)) {
            break;
        }

        /* Apply damped Newton step with floor clamping.
           fac ramps from 1/careful_steps to 1 over the first careful_steps iterations,
           preventing large overshoot in stiff regimes. */
        const double fac = fmin(1, ((float)num_iter + 1) / careful_steps);
        sv->T = fmax(All.MinGasTemp, fac * dsv.T + sv->T);
        sv->u = fmax(All.MinEgySpec * UNIT_SPECEGY_IN_CGS, fac * dsv.u + sv->u);
        num_iter++;

        if (solve_failure_condition(sv, &dsv, tol, num_iter)) {
            if (careful_steps == 1) {
                /* Attempt 1 failed: retry with 30-step damped ramp */
                careful_steps = 30;
                *sv = sv0;
                for (int i = 0; i < N_VARS; i++) dsv.data[i] = MAX_REAL_NUMBER;
                num_iter = 0;
                continue;
            } else if (careful_steps == 30) {
                /* Attempt 2 failed: bisection restart from geometric mean of T bounds */
                careful_steps = 31;
                sv->T = sqrt(T_upper * T_lower);
                sv->u = sv->T / U_TO_TEMP_UNITS;
                for (int i = 0; i < N_VARS; i++) dsv.data[i] = MAX_REAL_NUMBER;
                num_iter = 0;
                continue;
            } else {
                /* Attempt 3 failed: check if residual is small enough to accept.
                   Near the temperature floor the Jacobian becomes ill-conditioned,
                   producing huge dx even when the residual is effectively zero.
                   We use a relaxed tolerance (100x nominal) and warn. */
                double scale_T = fabs(pr->u_0 * pr->n_Htot / pr->Δt) + 1e-30;
                double scale_u = fabs(sv->u) + 1e-30;
                double relaxed_tol = 100 * tol;
                if ((fabs(func.T) < relaxed_tol * scale_T) && (fabs(func.u) < relaxed_tol * scale_u)) {
                    printf("WARNING: jaco accepting relaxed tolerance (%.0e vs nominal %.0e): n=%g T=%g u=%g resid_T=%g resid_u=%g\n",
                           relaxed_tol, tol, pr->n_Htot, sv->T, sv->u, fabs(func.T)/scale_T, fabs(func.u)/scale_u);
                    break;
                }
                printf("jaco failed to converge for num_iter=%d n=%g u0=%g u=%g T=%g X0=%g %g X=%g %g dx=%g %g func=%g "
                       "%g careful_steps=%d",
                       num_iter, pr->n_Htot, pr->u_0, sv->u, sv->T, sv0.T, sv0.u, sv->T, sv->u, dsv.T, dsv.u, func.T,
                       func.u, careful_steps);
                endrun(10);
            }
        }
    }
    return num_iter;
}
#endif
