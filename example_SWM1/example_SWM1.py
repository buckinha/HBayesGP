import SWMv1_3 as SWM
import HBayesGP, random
import numpy as np

def SWM_sample(policy):
    """Gets a single SWM v1.3 "sample"

    For SWMv1_3, we'll draw 10 monte carlo simulations at the given policy, which
    is the gaussian process "coordinate", and take the average value of all ten,
    and compute the variance 

    Gets simulation results and returns them in the form [X, y, y_var]

    PARAMETERS
    ----------
    policy: a list represnting the policy vector, which is the same thing as a 
        coordinate for the gaussian process. For SWM v1.3, this should be a 
        vector with length = 2 

    RETURNS
    -------
    X: the policy gven
    y: the average value of these simulations
    var: the variance of the values of these simulations
    """

    #SWM.simulate() function signature is:
    #simulate(timesteps, policy=[0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True

    sims = [SWM.simulate(200, policy, random.random(), SILENT=True) for i in range(10)]

    data = [  sims[i]["Average State Value"]  for i in range(10) ] 

    #now get mean and variance
    y = np.mean(data)
    var = np.var(data)

    return policy, y, var


def main():
    #get an initial set of samples
    # SWM v1.3 is a 2-dimensional model, so coordinates will be length-2 vectors
    X = []
    y = []
    y_var = []
    for i in range(25):
        coord = [random.uniform(-25,25), random.uniform(-25,25)]
        new_X, new_y, new_var = SWM_sample(coord)
        X.append(new_X)
        y.append(new_y)
        y_var.append(new_var)

    bounds = [[-25,25],[-25,25]]

    #instantiate a HBayesGP object
    gp = HBayesGP.HBayesGP(X,y,y_var,bounds)

    for i in range(20):
        #get a few suggestions, and sample from them
        next_coords = gp.suggest_next(number_of_suggestions=1)
        
        #check if suggest succeeded
        if len(next_coords) == 0:
            pass
        else:
            new_X, new_y, new_var = SWM_sample(next_coords[0])
            gp.add_data([new_X], [new_y], [new_var])


    #look at the best values it found
    gp.print_best_to_date()




if __name__ == '__main__':
    main()