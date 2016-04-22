# HBayesGP
An implementation of Bayesian optimization with point-wise variances.

###UNFINISHED WORK
This code is in an early stage, and is not yet suitable for use for opimization.

###Primary Features
HBayesGP uses a Gaussian process model to approximate the response surface of a simulator or experiment. Where most bayesian optimization routines assume each experiment/data point to represent a single value, HBayesGP allows each point to itself be the combination of many simulations or experiments. It will therefore accept both a value (assumed to be the mean) AND a variance estimate for each point.
During optimization, the algorithm will take into consideration both the GP confidence (derived from the process's covariance function) AND the simulator/experimental confidence, derived from the given variance estimates. Penalties on simulator/experimental variance can be applied in order to find local optima with tighter simulation variance, even if the mean of those simulations or experiments are actually less than another. This is of use when the downside risk of underperformance is more important than simply having a slightly better, overall expected value.
