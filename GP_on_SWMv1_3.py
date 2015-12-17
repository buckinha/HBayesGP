
import Queue, random, sys
sys.path.append("/home/hailey/Documents/FireGirl/gravity")
import SWMv1_3 as SWM
from sphere_dist import sphere_dist
import numpy as np
from sklearn import gaussian_process
from scipy.optimize import minimize
import scipy.spatial.distance

#OPTIONS

#how many "top values" to record at a time. I.e if set to 3, the top 3 values
# will be recorded throughout
best_values_count = 10

#how many individual SWM simulations should be averaged together for an 
# individual "sample" point
simulations_per_sample = 10

#how many samples for the initial set
initial_sample_size = 10

#how many independent hill-climbs to find new sampling positions
hill_climb_count = 1

#what radius should the "sphere of interest" be around the origin. No sampling
# will take place outside of this sphere.
sampling_radius = 30

#maximum number of times the GP loop can run, regardless of all other considerations
fail_safe_count = 100

#how close must samples be in order to be considered redundant
redundancy_dist = 0.4

#if the GP's predicted variance is too low at a given point, it will not be sampled.
# what value should that cutoff point be?
MSE_cutoff = 0.01 #???

bounds=[[-30,30],[-30,30]]

#global best value queue
best_values_queue = Queue.PriorityQueue()

def main():
    ### CONSTRUCT BEST VALUES QUEUE ###
    
    #initialize the queue so that we don't have to check it's size every time.
    for i in range(best_values_count):
        best_values_queue.put(float("-inf"))

    ### CONSTRUCT INITIAL SET OF SAMPLES ###
    #first, get some starting policies within the sphere of interest
    center=[0,0]
    radius=sampling_radius
    initial_policies = [sphere_dist(center, radius, random_seed=random.random()) for i in range(initial_sample_size)]

    #trim any redundant starting points
    initial_policies = trim_redundant(initial_policies)

    #now gather a set of y and X values for each of these policies
    #note that the get_simulation calls will return tuples of the form (y, var, X)
    data = [ get_simulations(count=simulations_per_sample,policy=pol) for pol in initial_policies ]

    #now look through each and get the best values
    for d in data: record_if_better(d[0])


    ### GP-FITTING LOOP ###
    """
    Forming the GP
        1) Transform the point-wise variances to fit the nugget properly
        2) Fit a GP with the nugget
        3) Draw some number of start positions for gradient descent
        4) Run L-BFGS-B from each of these start points to find new sampling spots
        5) Eliminate new sampling locations that are redundant
        6) Eliminate sampling points whose predicted variation is lower than some threshold
        STOPPING CRITERIA:
           Stop if step 6 eliminated all sampling locations.
        6) Draw new sample sets at each of these locations and add them to the data set
        7) Check if any of the newly sampled pathway sets are new maxima (or in the top 3, etc...)
        STOPPING CRITERIA:
           Stop if no new best has been found after some number of loops
           Stop if the maximum number of total simulations has been reached (if any is max is enforced)
        8) REPEAT
    """

    #instantiate the GP object
    gp = gaussian_process.GaussianProcess(theta0=1e-2)

    #prepare the X, y, and var arrays
    X = [0.0] * len(data)
    y = [0.0] * len(data)
    var = [0.0] * len(data)
    for d in range(len(data)):
        X[d] = data[d][2]
        y[d] = data[d][0]
        var[d] = data[d][1]

    term_message = "Failsafe iterations reached."

    #starting the main GP-fitting loop
    for outer_loop_i in range(fail_safe_count):

        #transform the variances into a nugget vector
        #according to the sklearn documentation, when using the squared exponential covariance 
        # function, the nugget should be equivalent to (var_i / y_i)^2 
        nugget = [ ( var[i] / y[i] ) * ( var[i] / y[i] ) for i in range(len(y)) ]


        #fit the GP
        #Note: from the sklearn.gaussian_process source, it looks like it is fine to reset the
        #nugget manually, i.e. gp.nugget = [whatever], rather than having to reinstantiate
        # the gp object each time in order to pass one in through the constructor
        gp.nugget = nugget[:]
        gp.fit(X, y)


        #draw hill-climb start points
        pts = [sphere_dist([0,0], sampling_radius, random_seed=random.random()) for i in range(hill_climb_count)]

        #do hill-climbs
        term_points = [ minimize(gp_variance, x0=pts[i], args=gp, method='L-BFGS-B', bounds=bounds).x for i in range(hill_climb_count)]

        #eliminate redundant termination points
        term_points = trim_redundant(term_points)

        #and eliminate any termination points that have already been sampled
        term_points = trim_redundant_from_comparison(term_points,X)


        #eliminate samples with variance below the cut-off point
        final_points = [[0.0,0.0] for i in range(len(term_points))]
        added = 0
        for i in range(len(term_points)):
            _val, MSE = gp.predict(term_points[i],eval_MSE=True)
            if MSE >= MSE_cutoff:
                final_points[added] = term_points[i]
                added += 1


        #check if any termination points remain
        if added == 0:
            #all of the non-redundant sampling points had mean squared errors below the cutoff point,
            # so there is nothing to sample
            term_message = "No hill-climb termination points had GP MSE above the cutoff point of " + str(MSE_cutoff)
            break


        #draw new samples from each remaining termination point
        new_data = [ get_simulations(count=simulations_per_sample,policy=pol) for pol in final_points ]

        #check the new samples for best-values
        for d in new_data: record_if_better(d[0])

        #add the new samples to the data set
        X +=   [d[2] for d in new_data]
        y +=   [d[0] for d in new_data]
        var += [d[1] for d in new_data]


        #check convergence criteria
        #TODO


    #exited the loop
    print("")
    print("GP-fitting Complete")
    print(term_message)
    print("Best Values Found")
    for i in range(best_values_count):
        print(str(best_values_queue.get()))

    print("")
    print("DATA SET")
    for i in range(len(X)):
        print("y = " + str(y[i]) + "   X = " + str(X[i]))



#Add this value to the list of top values if it is high enough, and retain order
def record_if_better(value):
    #add this value to the priority queue
    best_values_queue.put(value)

    #remove the lowest of these best values to keep the list the same length
    throw_away = best_values_queue.get()


#get a set of SWM simulations on one policy, reduced to relevant data: y, var, X
def get_simulations(count, policy, years=200):
    """Gets simulation results and returns them in the form [[y,X],...]

    ARGUEMENTS
    count: integer: how many simulations to run per policy given

    policy: list: the policy on which to simulate

    RETURNS y, var, X
    y: the average value of these simulations
    var: the variance of the values of these simulations
    X: the policy gven
    """

    #SWM.simulate() function signature is:
    #simulate(timesteps, policy=[0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True

    sims = [SWM.simulate(years, policy, random.random(), SILENT=True) for i in range(count)]

    data = [  sims[i]["Average State Value"]  for i in range(count) ] 

    #now get mean and variance
    y = np.mean(data)
    var = np.var(data)

    return y, var, policy

#Unused? Strip simulation dictionaries down to just their y and X vectors
def sim_reduce_down(simulaton_dict):
    """Reduces a SWM simulation from its dictionary form to a list [y, X]"""
    y = simulation_dict["Average State Value"]
    X = simulation_dict["Generation Policy"]
    return [y, X]

#Eliminate redundant points and return the trimmed list
def trim_redundant(X):
    #redundant values will be marked as False, not including the first occurance in the list.
    redundancy_mask = [True] * len(X)
    for i in range(len(X)-1):
        for j in range(i+1,len(X)):
            #skip this one if it's already been marked as redundant
            if redundancy_mask[j]:
                if scipy.spatial.distance.euclidean(X[i], X[j]) < redundancy_dist:
                    redundancy_mask[j] = False

    #build the trimmed list
    trimmed_list = [[0.0]*len(X[0]) for i in range(len(X))]
    added = 0

    for i in range(len(X)):
        if redundancy_mask[i]:
            #if this point isn't marked as redundant, add it to the trimmed list, and 
            # increment the index
            trimmed_list[added] = X[i][:]
            added += 1

    #return the portion of trimmed_list that has real values in it
    return trimmed_list[:added]

#trim from one list if they are redundant compared to another list
def trim_redundant_from_comparison(x_to_trim, x_to_compare):
    #for each value in x_to_compare, check if any of the x_to_trim
    # values are too close

    #create a mask
    keep_mask = [True] * len(x_to_trim)

    for i in range(len(x_to_compare)):
        #check if any of the x_to_trim points are too close to this x_to_compare point
        for j in range(len(x_to_trim)):
            #has this point already been trimmed?
            if keep_mask[j]:
                #it hasn't already been trimmed
                #should it be?
                if scipy.spatial.distance.euclidean(x_to_trim[j], x_to_compare[i]) < redundancy_dist:
                    #it is too close to one of the points in x_to_compare, so mark it for trimming
                    keep_mask[j] = False 
                    break

    #build the trimmed list
    trimmed_list = [[0.0]*len(x_to_trim[0]) for i in range(len(x_to_trim))]
    added = 0
    for i in range(len(x_to_trim)):
        if keep_mask[i]:
            #if this point isn't marked as redundant, add it to the trimmed list, and 
            # increment the index
            trimmed_list[added] = x_to_trim[i][:]
            added += 1

    #return the portion of trimmed_list that has real values in it
    return trimmed_list[:added]


#get a GP's variance at point x
def gp_variance(x,gp_object):
    FLIP_SIGN = True
    y, MSE = gp_object.predict(x, eval_MSE=True)
    if FLIP_SIGN:
        MSE *= -1.0
    return MSE


if __name__ == '__main__':
    main()