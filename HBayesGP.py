import Queue, random, sys
import numpy as np
from sklearn import gaussian_process
from scipy.optimize import minimize
import scipy.spatial.distance
from utils.sphere_dist import sphere_dist
from utils.Errors import InputError

class HBayesGP:
    """
    The primary HBayes Gaussian Process Optimization class.
    """
    def __init__(self,X,y,y_var=None,bounds=None):

        
        self.redundancy_dist = 0.2
        self.VERBOSE = False

        #the sklearn gp implementation does it's squared exponential covariance function
        # as 
        #                                   n
        # theta, d --> r(theta, d) = exp(  sum  -1 * theta_i * (d_i)^2 )
        #                                 i = 1
        #
        #
        #whereas what I want to work with is:
        #                                   n
        # theta, d --> r(theta, d) = exp(  sum  -1 * (1 / (2 * l^2) * (d_i)^2 )
        #                                 i = 1
        #
        #
        #So then, for a characteristic length-scale 'l', as per Rasmussen and Williams,
        #I just need to calculate theta as:
        # 
        # theta = (1 / (2 * l^2)
        self.gp_length_scale = 0.2
        self.gp_theta0 = (1.0 / (2.0 * self.gp_length_scale**2))
        #and set some reasonable bounds on what the length scale could possibly be
        #hmm... but this will make them switch places, ordinally, so I'll input them 
        # in reverse
        #self.gp_theta_UB = (1.0 / (2.0 *  2.0  **2))
        #self.gp_theta_LB = (1.0 / (2.0 *  0.01 **2))
        self.gp_theta_LB = (1.0 / (2.0 *  2.0  **2))
        self.gp_theta_UB = (1.0 / (2.0 *  0.01 **2))

        #and instantiate the gp
        self.gp = gaussian_process.GaussianProcess(theta0=self.gp_theta0, thetaL=self.gp_theta_LB, thetaU=self.gp_theta_UB, random_start=3)


        #instantiate the data set
        self.X = []
        self.y = []
        self.y_var = []
        self.nugget = []
        #everything that add_data needs should now be instantiated
        #but send a warning shot that this is the first time, so that add_data doesn't
        #try to record the best values to the (as-of-yet non-existant) priority queue
        self._NO_RECORDING_BEST = True
        self.add_data(X,y,y_var)
        self._NO_RECORDING_BEST = False

        #the boundaries on values for the elements of x coordinates
        # these are used both for drawing points within the region of interest,
        # and for limiting the range of the l-bfgs-b algorithm when it is 
        # searching for things in the data space.
        # they should have the form [[lb,ub],...,[lb,ub]] with lb and ub representing
        # the lower and upper bounds of the same-indexed element of any x vector
        if not bounds:
            self.bounds = [  [float("-inf"), float("inf")] for i in range(len(self.X[0])) ]
        else:
            self.bounds = bounds


        #start the global bests list
        self.global_bests_count = min(10, len(self.y))
        self.global_bests = Queue.PriorityQueue()
        #loop over the data set and fill the queue, since we skipped this before
        for i in range(len(self.y)):
            self._record_if_better(self.y[i],self.y_var[i],self.X[i])


    ### PUBLIC FUNCTIONS ###

    #use the gp to predict the value of y at coordinates x
    def predict(x):
        """ Predicts the y value and variance from the gaussian process at coordinate x

        PARAMETERS
        ----------
        x: the coordinate (explanatory/independent variables) at which to make a prediction

        RETURNS
        -------
        y_val: the predicted y value. This is the mean of the gaussian process at this x
            coordinate.

        MSE: the mean squared error of teh gaussian process at this x coordinate. Upper and
            lower confidence intervals will be approximately (y_val +/- 1.96*MSE)

        """

        #check if there's any data yet
        if len(self.X) == 0:
            if self.VERBOSE:
                print("Cannot make predictions without specifying fit data. Try HBayesGP.add_data()")
            return 0, 0
        
        y_val, MSE = self.gp.predict(x, eval_MSE=True)

        return y_val, MSE

    #add data to the model and re-fit the GP
    def add_data(self, X, y, y_var=None, NO_FIT=False):
        """ Adds data to the underlying gaussian process and re-fits

        This function takes the given X, y, and y-variance data and adds them to
        the dataset. It will check for redundant x coordinates (which are problematic)
        for the underlying sklearn.gaussian_process.GaussianProcess object and 
        remove/correct them. If all goes well, it will then re-fit the gaussian 
        process itself, so that predictions, plots, etc... can be run immediately

        PARAMETERS
        ----------
        X: A list of x coordinates (the explanatory variables) for each data point.

        y: A list of y values (the dependent variable) for each data point.

        y_var: A list of the variances associated with the y values for each data point.
            If no values are given, the variances are assumed to be close to machine 
            zero.

        NO_FIT: A boolean flag, default to False, indicating whether or not the fitting
            process should be skipped. By default, (False) the GP will always be re-fit
            after recieving new data.

        RETURNS
        -------
        None

        """
        #check lenghts of the inputs
        input_length = len(X)
        if not (   (len(X) == len(y)) and (len(y) == len(y_var))   ):
            #this is an error condition. All of the inputs must have equal length.
            expr = "HBayes.GP.add_data(self, X, y, y_var=None, NO_FIT=False)"
            msg = "The lengths of the input arguments X, y, and y_var must be the same."
            detail =  "len(X) = {0}, len(y) = {1}, len(y_var) = {2}".format(len(X), len(y), len(y_var))
            msg = msg + "\n" + detail
            raise InputError(expr, msg)

        #instantiate lists to recieve vetted inputs
        #the elements are set to the length of the appropriate lists/values
        #the lengths are set to the maximum
        X_remaining = [ [0.0]*len(X[0]) for i in range(input_length) ]
        y_remaining = [0.0] * input_length
        y_var_remaining = [0.000000001] * input_length
        added = 0

        #look through each X and see if it's too close to something already in self.X
        for i in range(input_length):
            if not self._coord_is_redundant(coord=X[i], compare_to_these=self.X, distance=self.redundancy_dist):
                X_remaining[added] = X[i]
                y_remaining[added] = y[i]
                #y_var may or may not be specified.
                if y_var:
                    y_var_remaining[added] = y_var[i]
                added += 1

        #Add any remaining values
        if not added == 0:
            self.X += X_remaining[:added]
            self.y += y_remaining[:added]
            self.y_var += y_var_remaining[:added]

            #and unless specified otherwise, re-fit the gp.
            if not NO_FIT:
                self._fit()

        else:
            if self.VERBOSE:
                print("WARNING: HBayes.GP.add_data():")
                print("  All additional inputs were redundant with previous inputs.")
                print("  No new data were added to the model.")

        #and now, if the priority queue has been instantiated, see if any of these values are global bests
        if not self._NO_RECORDING_BEST:
            #loop over the data set and fill the queue
            for i in range(added):
                self._record_if_better(y_remaining[i], y_var_remaining[i], X_remaining[i])

    #Suggests coordinates where samples should next be drawn.
    def suggest_next(self,number_of_suggestions, minimum_climbs=4):
        """ Suggests coordinates where samples should next be drawn.

        Using the chosen (or default) sampler method, this heuristic will
        look for likely coordinates in the gaussian process where new data 
        points should be sampled. Ideally, it will choose locations that will
        best improve the over-all fit of of the GP.

        TODO: implement samplers
        FOR NOW: the default sampler will be to look for local optima in the
        mean + 2 * variance of the GP, which is roughly equivalent to finding
        local maxima in the upper 95 percent confidence bound.

        the scipy.minimize, l-bfgs-b gradient decent routine will be used to find
        maxima, from a number of starting locations equal to the number of number
        of suggestions requested. However, a hard minimum number of hill climbs 
        can also be specified so that when only a few suggestions are desired,
        multiple climbs can still be conducted to find the best candidates.

        Additionally, the top data points (i.e. those with the best values) will
        also have hill climbs started from them.

        PARAMETERS
        ----------
        number_of_suggestions: An integer specifying how many suggested points
            should be returned.

        minimum_climbs: An integer (default = 4) specifying how many climbes should
            be attempted even if fewer suggestions are requested.


        RETURNS
        -------
        suggestions: a list of coordinates of length equal to 'number_of_suggestions'
            OR LESS. It is possible that the algorithm will not be able to find 
            enough suggestions given the starting points, in which case, a warning 
            will be printed if VERBOSE is set. Check the length of the return list
            to be sure.

        """

        #instantiate a list to hold valid suggestions
        suggestions = []

        #do climbs from global bests
        #accessing the PriorityQueue.queue gets the underlying array, so that I can
        # iterate over them in place (but un-ordered) without popping any off.
        for g_best in self.global_bests.queue:

            #do a climb from g_best
            ending_point = minimize(self._max_conf, x0=g_best[2], method='L-BFGS-B', bounds=self.bounds).x
            #add the result.x to the list
            suggestions.append(ending_point)


        #do climbs from random starting points
        climb_count = max(minimum_climbs, number_of_suggestions)
        for i in range(climb_count):

            #draw a random starting point
            starting_point = self._rand_coord()

            #do the climb
            ending_point = minimize(self._max_conf, x0=starting_point, method='L-BFGS-B', bounds=self.bounds).x

            #add the result.x to the list
            suggestions.append(ending_point)


        #eliminate redundant suggestions
        suggestions = self._trim_redundant(suggestions, self.redundancy_dist)

        #eliminate suggestions already represented in the data set
        suggestions = self._trim_redundant_from_comparison(suggestions, self.X, self.redundancy_dist)

        #are there enough suggestions still?
        if len(suggestions) < number_of_suggestions:
            if self.VERBOSE:
                #not enough suggestions, so say so
                print("WARNING: HBayesGP.suggest_next() could not find the full number of")
                print("  suggestions requested. Returning a total of {0}".format(len(suggestions)))


        #and finally, if the suggestions list is too long...
        #make a priority queue with the gp upper confidence as the prioirty
        if len(suggestions) > number_of_suggestions:
            sugg_pq = Queue.PriorityQueue()
            for sugg in suggestions:
                sugg_pq.put([self._max_conf(sugg), sugg[:]])

            #now pull values off until the length is equal to what we want
            while sugg_pq.qsize() > number_of_suggestions:
                throw_away = sugg_pq.get()

            #and finally, re-build the suggestions list
            suggestions = []
            while not sugg_pq.empty():
                suggestions.append(sugg_pq.get()[1][:])


        return suggestions

    def print_best_to_date(self):
        #TODO, fix this please (really messy output)
        print("Printing best values found so far:")
        for g_best in self.global_bests.queue:
            print(str(g_best))
        
    def plot_gp(self, dim0_index=0, dim1_index=1):
        """ Short Description

        Long Description

        PARAMETERS
        ----------

        RETURNS
        -------

        """
        pass
        
    def plot_gp_to_file(self, filename, dim0_index=0, dim1_index=1):
        """ Short Description

        Long Description

        PARAMETERS
        ----------

        RETURNS
        -------

        """
        pass
        
    def set_auto_sampler(self,sampler_function):
        """ Short Description

        Long Description

        PARAMETERS
        ----------

        RETURNS
        -------

        """
        pass
        
    def auto_next(self,number_of_new_samples=1):
        """ Short Description

        Long Description

        PARAMETERS
        ----------

        RETURNS
        -------

        """
        pass
        
    def save_data_set(self,filename):
        """ Short Description

        Long Description

        PARAMETERS
        ----------

        RETURNS
        -------

        """
        pass
        
    def load_data_set(self,filename):
        """ Short Description

        Long Description

        PARAMETERS
        ----------

        RETURNS
        -------

        """
        pass
        
    def set_suggest_method(self,method_name):
        """ Short Description

        Long Description

        PARAMETERS
        ----------

        RETURNS
        -------

        """
        pass
        


    ## PRIVATE FUNCTIONS ###

    #Fits the gp
    def _fit(self):
        """ Fits a gaussain process to the current data.

        Prepares the nugget from self.y_var, and fits the sklearn gp. According to 
        the documentation with sklearn.gaussian_process, the value of the nugget 
        for sample 'i' should be:   (var_i / y_i)^2
        when the squared exponential covariance function is used.

        PARAMETERS
        ----------
        None

        RETURNS
        -------
        None

        """
        
        #prepare the nugget array
        self.nugget = [0.0] * len(self.y)

        #do the calculations. According to the documentation with sklearn.gaussian_process,
        # the value of the nugget for sample i should be (var_i / y_i)^2
        for i in range(len(self.y)):
            self.nugget[i] = (self.y_var[i] * self.y_var[i]) / (self.y[i] * self.y[i])

        #set the nugget
        self.gp.nugget = self.nugget[:]

        #fit the underlying sklearn gp
        self.gp.fit(self.X, self.y)

    #Add this value to the list of top values if it is high enough, and retain order
    def _record_if_better(self,y,y_var,x):
        #add this value to the priority queue
        self.global_bests.put([y,y_var,x])

        #remove the lowest of these best values to keep the list the same length
        if self.global_bests.qsize() >= self.global_bests_count:
            throw_away = self.global_bests.get()

    #Determine whether a coordinate close to this one already exists in a set
    def _coord_is_redundant(self, coord, compare_to_these, distance):
        """ Determines whether the given coordinate is close to any in the given list.

        Checks to see if the given coordinate is within a certain distance from any 
        of the coordinates in the list provided.

        PARAMETERS
        ----------
        coord: the coordinate in question

        compare_to_these: a list of coordinates to compare to

        distance: the distance at which, or within, two coordinates are considered the same

        RETURNS
        -------
        True if 'coord' is within 'distance' of any of the coordinates in c'ompare_to_these'
        False otherwise.

        """

        for c in compare_to_these:
            if scipy.spatial.distance.euclidean(coord, c) < distance:
                return True

        #we checked all of them and nothing returned True, so coord is far enough away from all of them.
        return False

    #Eliminate redundant points and return the trimmed list
    def _trim_redundant(self, X, distance):
        """
        Trims a list of coordinates down to those that are unique within a given distance

        The function checks each pair of coordinates in the input list and determines
        if any two (or more) are closer than the given distance. If so, only one of them
        is kept, and the rest are removed from the list. Finally, a shortenned list is
        returned containing only points which are farther from each other than the given
        distance.

        Note that the processing order matters. For instance, if A is close to B, and B is
        close to C, but A is not close to C, then the function may return either a set
        containing only A, only C, or both A and C. 


        PARAMETERS
        ----------
        X:  A list of coordinates. Coordinates should be as lists, tuples, etc...

        distance:  The euclidean distance between two coordinates at which or below, they
            are considered to be equal


        RETURNS
        -------
        A list of coordinates which are each separated from each other by at least the 
            euclidean distance given.

        """
        #redundant values will be marked as False, not including the first occurance in the list.
        redundancy_mask = [True] * len(X)
        for i in range(len(X)-1):
            for j in range(i+1,len(X)):
                #skip this one if it's already been marked as redundant
                if redundancy_mask[j]:
                    if scipy.spatial.distance.euclidean(X[i], X[j]) < self.redundancy_dist:
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

    #Trim from one list if they are redundant compared to those in another list
    def _trim_redundant_from_comparison(self, x_to_trim, x_to_compare, distance):
        """
        Trims a list of coords to those which are far enough from coords in another list.

        The function checks each coordinate in the input list against each coordinate in 
        the comparison list. If an input coordinate is within the given distance of any
        of the comparison coordinates, it is removed from the list. Finally, the trimmed
        list is returned.

        The return list will only contain coordinates from the x_to_trim list.


        PARAMETERS
        ----------
        x_to_trim:  A list of coordinates to trim. Coordinates should be as lists, tuples,
            etc...

        x_to_compare: A list of coordinates against which to compare. These are not altered 
            or returned.

        distance:  The euclidean distance between two coordinates at which or below, they
            are considered to be equal.


        RETURNS
        -------
        A list containing a subset of the coordinates from x_to_trim, none of which is 
            closer than the given distance to any coordinate in x_to_compare.

        """


        #create a mask
        keep_mask = [True] * len(x_to_trim)

        for i in range(len(x_to_compare)):
            #check if any of the x_to_trim points are too close to this x_to_compare point
            for j in range(len(x_to_trim)):
                #has this point already been trimmed?
                if keep_mask[j]:
                    #it hasn't already been trimmed
                    #should it be?
                    if scipy.spatial.distance.euclidean(x_to_trim[j], x_to_compare[i]) < self.redundancy_dist:
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

    #Get a random coordinate within the bounds
    def _rand_coord(self):
        """ Draw a uniform random coordinate vector from within the bounds that have been set.

        A uniform random number is drawn between lower_bound and upper_bound for each 
        component of an x vector. The length of the x vector will be retrieved from the
        current dat set

        PARAMETERS
        ----------
        None

        RETURNS
        -------
        A list of the same length as the x vectors in self.X

        """
        return [random.uniform(b[0],b[1]) for b in self.bounds]


    ## PRIVATE Chooser Functions ##

    #The upper 95 percent confidence limit at x
    def _max_conf(self, x):
        """Returns the upper 95 percent confidence limit at point x

        PARAMETERS
        ----------
        x: the x vector from which to measure

        RETURNS
        -------
        conf: the value of the upper 95 percent confidence limit

        """

        _y_val, _MSE = self.gp.predict(x,eval_MSE=True)

        return _y_val + 1.96 * _MSE
