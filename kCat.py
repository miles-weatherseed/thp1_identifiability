import pints
from matplotlib import pyplot as plt
import numpy as np
import crossPresModels as crossPresModels
from scipy.integrate import solve_ivp

class MultiplicativeGaussianLogLikelihood(pints.ProblemLogLikelihood):
    r"""
    Calculates the log-likelihood for a time-series model assuming a
    heteroscedastic Gaussian error of the model predictions
    :math:`f(t, \theta )`.

    This likelihood introduces two new scalar parameters for each dimension of
    the model output: an exponential power :math:`\eta` and a scale
    :math:`\sigma`.

    A heteroscedascic Gaussian noise model assumes that the observable
    :math:`X` is Gaussian distributed around the model predictions
    :math:`f(t, \theta )` with a standard deviation that scales with
    :math:`f(t, \theta )`

    .. math::
        X(t) = f(t, \theta) + \sigma f(t, \theta)^\eta v(t)

    where :math:`v(t)` is a standard i.i.d. Gaussian random variable

    .. math::
        v(t) \sim \mathcal{N}(0, 1).

    This model leads to a log likelihood of the model parameters of

    .. math::
        \log{L(\theta, \eta , \sigma | X^{\text{obs}})} =
            -\frac{n_t}{2} \log{2 \pi}
            -\sum_{i=1}^{n_t}{\log{f(t_i, \theta)^\eta \sigma}}
            -\frac{1}{2}\sum_{i=1}^{n_t}\left(
                \frac{X^{\text{obs}}_{i} - f(t_i, \theta)}
                {f(t_i, \theta)^\eta \sigma}\right) ^2,

    where :math:`n_t` is the number of time points in the series, and
    :math:`X^{\text{obs}}_{i}` the measurement at time :math:`t_i`.

    For a system with :math:`n_o` outputs, this becomes

    .. math::
        \log{L(\theta, \eta , \sigma | X^{\text{obs}})} =
            -\frac{n_t n_o}{2} \log{2 \pi}
            -\sum ^{n_o}_{j=1}\sum_{i=1}^{n_t}{\log{f_j(t_i, \theta)^\eta
            \sigma _j}}
            -\frac{1}{2}\sum ^{n_o}_{j=1}\sum_{i=1}^{n_t}\left(
                \frac{X^{\text{obs}}_{ij} - f_j(t_i, \theta)}
                {f_j(t_i, \theta)^\eta \sigma _j}\right) ^2,

    where :math:`n_o` is the number of outputs of the model, and
    :math:`X^{\text{obs}}_{ij}` the measurement of output :math:`j` at
    time point :math:`t_i`.

    Extends :class:`ProblemLogLikelihood`.

    Parameters
    ----------
    ``problem``
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem two parameters are added (:math:`\eta`,
        :math:`\sigma`), for a multi-output problem 2 times :math:`n_o`
        parameters are added.
    """

    def __init__(self, problem):
        super(MultiplicativeGaussianLogLikelihood, self).__init__(problem)

        # Get number of times and number of outputs
        self._nt = len(self._times)
        no = problem.n_outputs()
        self._np = 2  # 2 parameters added per output

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._np

        # Pre-calculate the constant part of the likelihood
        self._logn = 0.5 * self._nt * no * np.log(2 * np.pi)

    def __call__(self, x):
        # Get noise parameters
        eta = x[-2]
        sigma = x[-1]

        # Evaluate function (n_times, n_output)
        function_values = self._problem.evaluate(x[:-self._np])

        # mask the zeros
        function_values[function_values == 0] = np.nan

        # Compute likelihood
        log_likelihood = \
            -self._logn - np.nansum(
                np.nansum(np.log(function_values**eta * sigma), axis=0)
                + 0.5 / sigma**2 * np.nansum(
                    (self._values - function_values)**2
                    /function_values ** (2 * eta), axis=0))
        return log_likelihood

class inferenceModel(pints.ForwardModel):
    def __init__(self, mode=0):
        # give mode 0 for ERAP, mode 1 for no ERAP
        self.mode = mode
        return
    def n_parameters(self):
        return 5
    def n_outputs(self):
        return 7
    def simulate(self, params, times):
        params = np.multiply(np.asarray([2694.91, 400, 12, 0.0000027, 12, 400/60, 33/60, 58.8/60, 400/60, 400000/18000, 400000/18000, 400000/18000, 400000/18000, 400000/18000]), np.hstack([np.ones(9), params]))
        #y_ss = solve_ivp(crossPresModels.no_erap_ss, [0, 1000000], np.zeros(8), args = tuple(params), method='LSODA').y.T[-1, :].astype(int)
        #y0 = np.hstack([np.zeros(5), y_ss[0:4], np.zeros(5), y_ss[4], np.zeros(5), y_ss[5], np.zeros(5), y_ss[6:]])
        if self.mode == 0:
            y0_E = np.asarray([0, 0, 0, 0, 0, 195271, 2992, 29558, 17133, 0, 0, 0, 0, 0, 42, 0, 0, 0, 0, 0, 3893, 0, 0, 0, 0, 0, 48625, 52502])
            sol = solve_ivp(crossPresModels.erap_rhs_ivp, [times[0], times[-1]], y0_E, t_eval=times, args = tuple(params), method='LSODA')
        elif self.mode == 1:
            y0_nE = np.asarray([0, 0, 0, 0, 0, 508509, 2548, 29558, 5618, 0, 0, 0, 0, 0, 42, 0, 0, 0, 0, 0, 3324, 0, 0, 0, 0, 0, 48981, 52121])
            sol = solve_ivp(crossPresModels.no_erap_rhs_ivp, [times[0], times[-1]], y0_nE, t_eval=times, args = tuple(params), method='LSODA')
        else:
            return ValueError("Should be either 0 or 1 for mode!")
        return sol.y[21:, :].astype(int).astype(float).T
       
if __name__ == '__main__':
    default_params = np.ones(5)
    # mode = 1 because we want to fit supply rates to model without ERAP!
    model = inferenceModel(mode=1)
    times = 60*np.asarray([0, 15, 30, 60, 90, 120, 180, 240, 300, 360])#np.linspace(0, 360*60, 360)
    data = model.simulate(default_params, times)

    try:
        noisy_data = np.loadtxt('noisy_data0.txt')
    except:
        noisy_data = data + pints.noise.multiplicative_gaussian(1, 0.1, data)
        np.savetxt('data/supplyRatesData.txt', noisy_data)

    # fit to this data using MCMC

    default_params_noise = np.hstack([default_params, 1, 0.1]) # we add 2 unknowns (eta and sigma) for the multiplicative noise
    problem = pints.MultiOutputProblem(model, times, noisy_data)
    log_prior = pints.UniformLogPrior(pints.RectangularBoundaries(0.1*default_params_noise, 10*default_params_noise))
    log_likelihood = MultiplicativeGaussianLogLikelihood(problem)
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    mcmc = pints.MCMCController(log_posterior, 3, [default_params_noise, default_params_noise*0.8, default_params_noise*1.25], method=pints.HaarioBardenetACMC)
    mcmc.set_parallel(False)
    mcmc.set_max_iterations(2000)
    chains = mcmc.run()
    reps = 1
    while max(pints.rhat(chains[:, :, :])) > 1.10:
        mcmc = pints.MCMCController(log_posterior, 3, chains[:, -1, :], method=pints.HaarioBardenetACMC)
        mcmc.set_parallel(False)
        mcmc.set_max_iterations(2000)
        mcmc.set_log_to_screen(False)
        chain = mcmc.run()
        new_chains = np.zeros((chains.shape[0], chains.shape[1] + 2000, chains.shape[2]))
        new_chains[:, :-2000, :] = chains
        new_chains[:, -2000:, :] = chain[:, :, :]
        chains = new_chains
        reps += 1
        print(pints.rhat(chains[:, :, :]))

    for i in range(3):
        np.savetxt('results/supplyRates_' + str(i + 1), chains[i, :, :])
