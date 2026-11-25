"""Conditional autoregressive co-Kriging with a frozen low-fidelity GP."""

import numpy as np

from 分目标气动高斯过程 import feature_map, kernel_cross


def fit_autoregressive(low_gp, x_high, y_high, kernel="matern32"):
    from scipy.optimize import minimize
    x_high, y_high = np.asarray(x_high, float), np.asarray(y_high, float)
    if len(x_high) < 4 or y_high.shape != (len(x_high),):
        raise ValueError("At least four high-fidelity observations required")
    z = (feature_map(x_high, low_gp["features"])-low_gp["feature_mean"])/low_gp["feature_scale"]
    train = np.asarray(low_gp["train"])
    inverse_low = np.asarray(low_gp["inverse"])
    low_cross = kernel_cross(z, train, np.asarray(low_gp["length"]),
                             low_gp["amplitude"], low_gp["kernel"])
    low_mean = low_cross @ low_gp["dual"]
    low_cov = (kernel_cross(z, z, np.asarray(low_gp["length"]),
                            low_gp["amplitude"], low_gp["kernel"])
               - low_cross @ inverse_low @ low_cross.T)
    low_cov = (low_cov+low_cov.T)/2
    normalized = (y_high-low_gp["mean"])/low_gp["scale"]
    p = z.shape[1]

    def unpack(theta):
        return theta[0], theta[1], np.exp(theta[2]), np.exp(theta[3:3+p]), np.exp(theta[-1])

    def system(theta):
        rho, bias, amplitude, length, noise = unpack(theta)
        covariance = rho**2*low_cov+kernel_cross(z, z, length, amplitude, kernel)
        covariance += np.eye(len(z))*(noise+1e-8)
        residual = normalized-rho*low_mean-bias
        return covariance, residual

    def likelihood(theta):
        covariance, residual = system(theta)
        try:
            chol = np.linalg.cholesky(covariance)
            whitened = np.linalg.solve(chol, residual)
        except np.linalg.LinAlgError:
            return 1e30
        return float(.5*(whitened @ whitened)+np.log(np.diag(chol)).sum()
                     +.5*len(z)*np.log(2*np.pi))

    variance = max(float(np.var(normalized-low_mean)), .01)
    bounds = [(.1, 2.5), (-20., 20.), (np.log(.001), np.log(100.))]
    bounds += [(np.log(.05), np.log(50.))]*p+[(np.log(1e-8), np.log(1.))]
    fits = []
    for initial_length in (1., 3.):
        initial = np.r_[1., float(np.mean(normalized-low_mean)), np.log(variance),
                        np.full(p, np.log(initial_length)), np.log(.01)]
        fits.append(minimize(likelihood, initial, method="L-BFGS-B", bounds=bounds,
                             options=dict(maxiter=200, ftol=1e-9)))
    fitted = min(fits, key=lambda fit: fit.fun)
    covariance, residual = system(fitted.x)
    inverse_high = np.linalg.solve(covariance, np.eye(len(z)))
    dual = inverse_high @ residual
    rho, bias, amplitude, length, noise = unpack(fitted.x)
    return dict(
        kind="autoregressive-gp", low_gp=low_gp, high_x=x_high.tolist(),
        high_z=z.tolist(), low_cross_high=low_cross.tolist(),
        rho=float(rho), bias=float(bias), kernel=kernel,
        amplitude=float(amplitude), length=length.tolist(), noise=float(noise),
        dual=dual.tolist(), inverse=inverse_high.tolist(),
        combined_low_dual=(rho*np.asarray(low_gp["dual"])
                           -rho**2*inverse_low @ low_cross.T @ dual).tolist(),
        interval_scale=1.,
        fit_status=dict(success=bool(fitted.success), message=str(fitted.message),
                        negative_log_likelihood=float(fitted.fun)),
    )


def autoregressive_prediction(model, x, uncertainty=True):
    low = model["low_gp"]
    z = (feature_map(x, low["features"])-low["feature_mean"])/low["feature_scale"]
    high_z = np.asarray(model["high_z"])
    low_cross = kernel_cross(z, np.asarray(low["train"]), np.asarray(low["length"]),
                             low["amplitude"], low["kernel"])[0]
    low_to_high = kernel_cross(z, high_z, np.asarray(low["length"]),
                               low["amplitude"], low["kernel"])[0]
    delta = kernel_cross(z, high_z, np.asarray(model["length"]),
                         model["amplitude"], model["kernel"])[0]
    rho = model["rho"]
    mean = (model["bias"]+low_cross @ model["combined_low_dual"]
            +(rho**2*low_to_high+delta) @ model["dual"])
    mean = float(low["mean"]+low["scale"]*mean)
    if not uncertainty:
        return mean
    inverse_low = np.asarray(low["inverse"])
    conditional_cross = low_to_high-low_cross @ inverse_low @ np.asarray(model["low_cross_high"]).T
    cross = rho**2*conditional_cross+delta
    variance = (rho**2*max(0., low["amplitude"]-low_cross @ inverse_low @ low_cross)
                +model["amplitude"]+model["noise"]+1e-8
                -cross @ np.asarray(model["inverse"]) @ cross)
    return mean, float(low["scale"]*np.sqrt(max(0., variance))*model["interval_scale"])
