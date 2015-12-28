import SWMv1_3 as SWM1
import SWMv2_1 as SWM2
import HBayesGP, random
import numpy as np
from sklearn.ensemble import RandomForestClassifier

GOOD_POL = 0
LB_POL = 1
SA_POL = 2
OOB_POL = 3

class SWMAnalyst:
    def __init__(self):

        ## SAMPLING PARAMETERS ##
        self.sims_per_sample = 20
        self.USING_SWM2_1 = False

        ## OUT-OF-BOUNDS PARAMETERS ##
        #whether or not to filter out-of-bounds policies. If True, these policies
        #will never be sampled, but will be assigned default values as if they'd
        #been sampled and turned out badly.
        self.FILTER_OOB = True
        #the default values for out-of-bounds "samples"
        self.OOB_default_mean = 0.0
        self.OOB_default_var = 0.0001

        ## SA/LB FILTER PARAMETERS ##
        self.USING_CLASSIFIER = False
        self.classifier_type = "random forest"
        #the default values for the simlulation means of SA and LB sims will
        #be updated as new simulations are actually run. The values below are set
        #to start the process off.
        self.LB_default_mean = 5.0
        self.SA_default_mean = 5.0
        self.edge_default_var = 0.0001

        ## RANDOM FOREST CLASSIFIER PARAMETERS ##
        # will be used if classifier_type is set to "random forest"
        self.RFC_estimators = 40
        self.RFC_depth = 20

        ## Initilaize classifier ##
        self.classfier = None
        self._init_classifier()
        

        ## DATA ##
        self.X = []
        self.y = []
        self.y_var = []
        self.supprate = []
        self.classification = []

        ## BOUNDARIES ##
        self.bounds = []
        self.default_upper_bound_value =  25.0
        self.default_lower_bound_value = -25.0
        self.set_bounds()

        ## DATA GENERATION PARAMETERS ##
        self.grid_sampling_density = 0.25

        ## GAUSSIAN PROCESS OBJECT ##
        #this will be instantiated the first time that data is recieved
        self.GP = None




    ### PUBLIC FUNCTIONS

    #Gets a single SWM v1.3 or SWMv2.1 "sample"
    def SWM_sample(self, policy):
        """Gets a single SWM v1.3 "sample"

        We'll draw monte carlo simulations at the given policy, which
        is the gaussian process "coordinate", and take the average value of all,
        and compute the variance and average suppression rate

        Gets simulation results and returns them in the form [y, y_var, X, supprate]

        The function will run SWMv1.3 samples unless the self.USING_SWM2_1 flag is 
        set to true, in which case SWMv2.1 samples will be used.

        The number of simulations per "sample" is set using self.sims_per_sample


        PARAMETERS
        ----------
        policy: a list represnting the policy vector, which is the same thing as a 
            coordinate for the gaussian process. For SWMv1.3, this should be a 
            vector with length = 2.  For SWMv2.1, this should be a length 6 vector


        RETURNS
        -------
        y: the average value of these simulations

        var: the variance of the values of these simulations

        X: the policy gven

        supprate: the average suppression rate of these simulations

        """

        
        sims = []
        if self.USING_SWM2_1:
            #SWMv2_1.simulate() function signature is:
            #def simulate(timesteps, policy=[0,0,0,0,0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True):
            sims = [SWM2.simulate(200, policy, random.random(), SILENT=True) for i in range(self.sims_per_sample)]
        else:
            #SWMv1_3.simulate() function signature is:
            #simulate(timesteps, policy=[0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True
            sims = [SWM1.simulate(200, policy, random.random(), SILENT=True) for i in range(self.sims_per_sample)]


        data = [  sims[i]["Average State Value"]  for i in range(self.sims_per_sample) ] 
        suppression_rates = [  sims[i]["Suppression Rate"] for i in range(self.sims_per_sample) ]

        #now get mean and variance
        y = np.mean(data)
        var = np.std(data)
        supprate = np.mean(suppression_rates)

        return y, var, policy, supprate

    #gets new samples from the simulators
    def add_samples(self, sample_count=1, policy_type="auto", policies=None):
        """Gets new samples from the simulators

        Uses whichever simulator is flagged (according to self.USING_SWM2_1) to 
        simulate new pathways and add them as samples to the current data set.
        The policies used to run each set of simulations is based on the policy_type
        arguement, which is by defualt set to "auto". 

        PARAMETERS
        sample_count: how many samples to add to the data set. Each sample will
            itself be based on the average values of some number of simulations, set
            by self.sims_per_sample.
            NOTE: sample_count will be ignored if policy_type is set to "grid"


        policy_type: a string indicating what policy-generating behavior to use.
            Options are: "auto", "random", "grid"


        RETURNS
        None
        """

        pol_len = 2
        if self.USING_SWM2_1: pol_len = 6


        #checking for an instantiated GP object
        if not self.GP:
            #the GP hasn't been instantiated, most likely because there's no data yet.
            #so set it up for a pass through the grid sampler
            if policy_type == "auto": policy_type = "grid"


        #generate policies
        pols = [ [0.0] * pol_len for i in range(sample_count)]
        if not policies:
            #no policies were given, so generate them
            if policy_type == "auto":
                #ask the gp object for it's next guesses
                #TODO
                pass
            elif policy_type == "grid":
                pols = self._grid_sampling_coordinates()
            else:
                #default to "random"
                for i in range(sample_count):
                    pols[i] = [  random.uniform(self.bounds[j][0], self.bounds[j][1]) for j in range(pol_len) ]
                
        else:
            #policies were given, so ignore policy_type, and assign them
            pols = policies[:]



        #get samples
        if len(pols) < 1:
            print("WARNING: No policies were generated...") #TODO add more information
        else:
            #loop over policies and get samples at each
            new_y = []
            new_y_var = []
            new_X = []
            new_supprate = []
            for p in pols:
                #check filters
                if (self.USING_CLASSIFIER) and (self.classify_policy(p) != GOOD_POL):
                    pass
                if self.FILTER_OOB:
                    pass

                _y, _var, _X, _supp = self.SWM_sample(p)
                new_y.append(_y)
                new_y_var.append(_var)
                new_X.append(_X)
                new_supprate.append(_supp)


            #add new data to the current data set
            self.y = self.y + new_y
            self.y_var = self.y_var + new_y_var
            self.X = self.X + new_X
            self.supprate = self.supprate + new_supprate

            #if there's already a GP object, then add data to it
            if self.GP:
                self.GP.add_data(new_X, new_y, new_y_var, NO_FIT=True)
            else:
                #there's no GP, so make one with ALL of the current data (not just the new ones)
                self.GP = HBayesGP.HBayesGP(self.X, self.y, self.y_var, self.bounds)


            #re-fit the GP and classifier models
            self.fit_GP() # <-- if there's no GP, then this function will add ALL the data to a new one.
            self.fit_classifier()


    #TODO
    def filter_edge_policies(self, new_policies):
        """Returns a list of True/False values reflecting whether each policy should be sampled.

        Summary
        Does predictions based on the underlying classification model and determines whether 
        each of the policies given is likely to be an edge case or not. A list of True/False 
        values are passed back, where True indicates that the associated policy SHOULD be sampled
        i.e. not an edge policy, and False indicates that the assoicated policy is likely to 
        be an edge policy and should NOT be sampled.

        PARAMETERS
        new_policies: a list of policies to be filtered

        RETURNS
        policy mask: a list of length equal to new_policies with True values at indicies where
        the policy in new_policies ought to be sampled (not an edge policy) and False where
        the policy in new_policies ought not to be sampled (a likely edge policy)

        """

        #TODO
        return []

    #TODO
    def fit_classifier(self):
        """Uses existing data to fit the edge-detection classifier.

        PARAMETERS
        None

        RETURNS
        None 
        """

        pass

    #TODO Check's a policy against the current classifier system
    def classify_policy(self, policy):
        #TODO
        #returns GOOD_POL, SA_POL, LB_POL, OOB_POL
        
        #first check OOB
        if self.check_for_OOB_policy(policy) == OOB_POL:
            return OOB_POL
        else:
            #policy is in-bounds; run the classfier
            #TODO!!!
            return GOOD_POL

    #TODO Check's if a policy is in or out of bounds
    def check_for_OOB_policy(self, policy):
        #is there data yet?
        if len(self.X) < 1:
            print("ERROR: there isn't any data yet")
            return OOB_POL

        for i in range(len(self.X[0])):
            if (policy[i] >= self.bounds[i][0]) and (policy[i] <= self.bounds[i][1]):
                pass
            else:
                return OOB_POL

        #we've completed the loop without returning OOB_POL, so this must be in bounds
        return GOOD_POL


    #Creates (if necessary) and fits a gaussian process to the data
    def fit_GP(self):
        """Creates (if necessary) and fits a gaussian process to the data

        PARAMETERS
        None 

        RETURNS
        None
        """

        #has a GP object been instantiated?
        if not self.GP:
            #nope. Time to make one.
            #TODO, if this is called before self.X, etc... have any data, I'm not sure what will happen
            self.GP = HBayesGP.HBayesGP(self.X, self.y, self.y_var, self.bounds)

        #and pass the fit command on down (ultimately to an sklearn.gaussian_process)
        self.GP._fit()

    #sets bounds to given, or to default values
    def set_bounds(self, new_bounds=None):
        lb = self.default_lower_bound_value
        ub = self.default_upper_bound_value
        if not new_bounds:
            if self.USING_SWM2_1:
                self.bounds = [[lb,ub],[lb,ub],[lb,ub],[lb,ub],[lb,ub],[lb,ub]]
            else:
                self.bounds = [[lb,ub],[lb,ub]]
        else:
            self.bounds = new_bounds[:]



    ### PRIVATE FUNCTIONS ###

    #instantiates the classifier object(s)
    def _init_classifier(self):
        if self.classifier_type = "random forest":
            self.classfier = RandomForestClassifier(max_depth=self.RFC_depth, n_estimators=self.RFC_estimators)
        else:
            print("ERROR: Unknown classifer value: " + str(self.classifier_type))

    #returns the coordinates for a grid sampling regime
    def _grid_sampling_coordinates(self):
        """Using grid_sampling_density and bounds, returns appropriate coordinates for sampling.

        RETURNS
        a list of coordinates(policies) representing the grid.

        """

        #for each dimension, get the number of segments into which to divide it.
        #the sampling density contained in self.grid_sampling_density indicate how far apart
        #  each point should be. Since these might not give rise to even multiples accross
        #  the range between each upper and lower bound, I'm doing a division and a ceiling 
        #  operation to ensure that sampling is the dense or better, but not worse.

        pol_len = 2
        if self.USING_SWM2_1: pol_len = 6

        slice_counts = [0] * pol_len
        for i in range(pol_len):
            slice_counts[i] = abs(self.bounds[i][0] - self.bounds[i][1]) / self.grid_sampling_density
            slice_counts[i] = np.ceil(slice_counts[i])

        #get slices
        slices = []
        for i in range(pol_len):
            slices.append(np.linspace(self.bounds[i][0], self.bounds[i][1], slice_counts[i]))


        #get coordinates as x and y vectors
        coordinate_list = []
        def _recur_get_next_line(slices, current_depth, current_coord, return_list):
            """ instantiate with current_depth = 0, current_coord is a zero vector, and
            return_list is an empty list"""

            #if we're not at the bottom
            if current_depth < len(slices) - 1:
                for _i in (slices[current_depth]):
                    current_coord[current_depth] = _i
                    return_list = _recur_get_next_line(slices, current_depth+1, current_coord, return_list)

                #done with this level, so back up with all the values which have been added to the return list
                return return_list

            else:
                #we are at the bottom
                
                #get the coordinates at every value at this depth, and add them to the return_list
                new_coords = [None] * len(slices[current_depth])
                for _i in range(len(slices[current_depth])):
                    current_coord[current_depth] = slices[current_depth][_i]
                    new_coords[_i] = current_coord[:]

                #concatenate the lists
                return_list = return_list + new_coords[:]

                return return_list

        _recur_get_next_line(slices, 0, [0.0] * pol_len, coordinate_list)


        return coordinate_list