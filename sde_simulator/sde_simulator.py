from functools import lru_cache

import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy.typing as npt


class SDESimulatorBase(ABC):
    """
    This is a base class for dynamic models, implementing a stochastic Rung-Kutta solver:
    y(t+1) = y(t) + a(y(t))*dt + b(y(t))*dW + 0.5*(b(y(t) + a(y(t)*dt + b(y(t)*sqrt(dt)) - b(Y(t))*(dW^2 - dt)/sqrt(dt)
    """

    @abstractmethod
    def __init__(self, y0: npt.NDArray, tau: [float, int] = 1., max_t: [float, int] = 1., dt: float = 1e-2,
                 noise: float = 0., n_sim: int = 10000, verbose: bool = True, **kwargs):
        """
        @param y0: Iterable convertible to numpy array. The initial value of the simulation variables.
                   Should be 1D of size #variables.
        @param tau: integer/float. The time constant of the simulation.
        @param max_t: integer/float. Determines the number of steps in the simulation together with dt.
        @param dt: float. The time between two steps in the simulation.
        @param noise: float. The standard deviation of the Wiener process
        @param n_sim: integer. The number of simulations to run.
        @param verbose: bool. Controls whether print_str prints the given argument or not.
        """
        self._verbose = verbose
        self._y0 = np.array(y0)
        self._tau = tau
        self._dt = dt
        self._time = np.arange(0, max_t + self._dt, self._dt)
        self._sqrtdt = np.sqrt(self._dt)
        # shape of (time, # variables, # simulations)
        self._y = np.zeros(self.time.shape + self.y0.shape + (n_sim,), dtype=float)
        self._y[0, :, :] = self._y0[:, None]
        self.print_str("generating noise sequences...")
        self._noise = noise
        self._generated_dw = np.random.normal(loc=0.0, scale=noise * self.sqrtdt, size=self.y.shape)
        self._generated_dw_squared = self._generated_dw ** 2
        self._simulated = False
        self._simulation_kwargs = dict()
        self.kwargs = kwargs

    def print_str(self, string):
        if self._verbose:
            print(string)

    @property
    def y0(self):
        return self._y0

    @property
    def y(self):
        return self._y

    @property
    def tau(self):
        return self._tau

    @property
    def dt(self):
        return self._dt

    @property
    def sqrtdt(self):
        return self._sqrtdt

    @property
    def time(self):
        return self._time

    @property
    def noise(self):
        return self._noise

    @lru_cache
    def dW(self, t):
        return self._generated_dw[t]

    @lru_cache
    def dWsquared(self, t):
        return self._generated_dw_squared[t]

    @abstractmethod
    def deterministic_func(self, y, i, **kwargs):
        pass

    @abstractmethod
    def stochastic_func(self, y, i, **kwargs):
        pass

    def _step(self, i, **kwargs):
        yn = self.y[i - 1, :]
        an = self.deterministic_func(yn, i=i, **kwargs) * (self.dt / self.tau)
        bn = self.stochastic_func(yn, i=i, **kwargs) / self.tau
        dw = self.dW(i)
        bnewn = self.stochastic_func(yn + an + bn * self.sqrtdt, i=i, **kwargs)
        self.y[i, ...] = yn + an + bn * dw + (0.5 / self.sqrtdt) * (bnewn - bn) * (self.dWsquared(i) - self.dt)

    def simulate(self, seed=97, use_tqdm=True, **kwargs):
        self._simulated = True
        self._simulation_kwargs.update(kwargs)
        np.random.seed(seed)
        loop_range = tqdm(range(1, self.time.size), desc="Simulating") if use_tqdm else range(1, self.time.size)
        for i in loop_range:
            self._step(i, **kwargs)
