#include "../allvars.h"
#include "indices.h"
#include "math.h"
#include "microphysics_func_jac.h"

int do_timestep(double *X, double *params) {
    double funcjac[NUM_VARS * (NUM_VARS + 1)], func[NUM_VARS], jac[NUM_VARS * NUM_VARS], jacinv[NUM_VARS * NUM_VARS],
        dx[NUM_VARS] = {1e100, 1e100};

    double tol = 1e-3;
    int num_iter = 0;
    while ((fabs(dx[0]) > tol * fabs(X[0])) && (fabs(dx[1]) > tol * fabs(X[1]))) {
        const int careful_steps = 1;
        microphysics_func_jac(X, params, funcjac);
        for (int i = 0; i < NUM_VARS; i++) {
            func[i] = funcjac[i];
        }
        for (int i = 0; i < NUM_VARS * NUM_VARS; i++) {
            jac[i] = funcjac[i + NUM_VARS];
        }
        double det = jac[0] * jac[3] - jac[1] * jac[2];
        jacinv[0] = jac[3] / det;
        jacinv[3] = jac[0] / det;
        jacinv[1] = -jac[1] / det;
        jacinv[2] = -jac[2] / det;
        dx[0] = -(jacinv[0] * func[0] + jacinv[1] * func[1]);
        dx[1] = -(jacinv[2] * func[0] + jacinv[3] * func[1]);
        if (X[INDEX_T] == All.MinGasTemp && dx[INDEX_T] < 0) {
            break;
        }
        // TODO: allow the full Newton step as long as it reduces the residual by at least half
        const double fac = fmin(1, ((float)num_iter + 1) / careful_steps);
        X[INDEX_T] = fmax(All.MinGasTemp, fac * dx[INDEX_T] + X[INDEX_T]);
        X[INDEX_u] = fmax(All.MinEgySpec, fac * dx[INDEX_u] + X[INDEX_u]);
        num_iter++;
        // if (params[IDX_n_Htot] < 1100 && params[IDX_n_Htot] > 1000)
        //     printf("num_iter=%d X=%g %g dx=%g %g func=%g %g\n", num_iter, X[0], X[1], dx[0], dx[1], func[0],
        //     func[1]);
        if (num_iter > MAXITER) {
            printf("jaco failed to converge for n=%g u=%g T=%g dx=%g %g func=%g %g", params[INDEX_n_Htot],
                   params[INDEX_u_0], X[INDEX_T], dx[0], dx[1], func[0], func[1]);
            endrun(10);
        }
    }
    //    printf("nH=%g num_iter=%d\n", params[IDX_n_Htot], num_iter);
    return num_iter;
}

void jaco_do_cooling(int i) {
    double Δt = GET_PARTICLE_TIMESTEP_IN_PHYSICAL(i) * UNIT_TIME_IN_CGS;
    double u_0 = SphP[i].InternalEnergy * UNIT_SPECEGY_IN_CGS, T = SphP[i].InternalEnergy * U_TO_TEMP_UNITS, u = u_0,
           n_Htot = nH_CGS(i), x_H = 1.,
           pdv_work = SphP[i].DtInternalEnergy; // * UNIT_SPECEGY_IN_CGS / UNIT_TIME_IN_CGS;
#include "assignments.h"                        // assign initial values of X and params
    int num_iter = do_timestep(X, params);
    SphP[i].InternalEnergy = X[INDEX_u] / UNIT_SPECEGY_IN_CGS;
}
