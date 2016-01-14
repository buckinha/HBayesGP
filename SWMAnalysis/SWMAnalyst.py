import SWMv1_3 as SWM1
import SWMv2_1 as SWM2
import HBayesGP
import random
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
        #using squared OOB distance penalty?
        self.USING_OOB_PENALTY = True
        #when using the squared OOB distance penalty, what is the scaling parameter
        self.OOB_alpha = 0.5

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
        self.in_bounds = [] #True for a sample that is in-bounds
        self.real_sample = [] #True for an actual sample, False for one that's been predicted

        ## BOUNDARIES ##
        self.bounds = []
        self.default_upper_bound_value =  25.0
        self.default_lower_bound_value = -25.0
        self.default_penalty_distance = 3
        self.penalty_bounds = []
        self.set_bounds()

        ## DATA GENERATION PARAMETERS ##
        self.grid_sampling_density = 6.0

        ## GAUSSIAN PROCESS OBJECT ##
        #this will be instantiated the first time that data is recieved
        self.GP = None




    ### PUBLIC FUNCTIONS ###

    #set the model to swm1 or swm2
    def set_SWM_model(self, model_integer):
        """Set to SWMv1.3 or SWMv2.1 by passing a 1 or a 2, respectively"""
        if model_integer == 2:
            self.USING_SWM2_1 = True
            self.grid_sampling_density = 20.0
            self.set_bounds()
        else:
            self.USING_SWM2_1 = False
            self.grid_sampling_density = 6.0
            self.set_bounds()

    #Gets a single SWMv1.3 or SWMv2.1 "sample"
    def SWM_sample(self, policy):
        """Gets a single SWMv1.3 or SWMv2.1 "sample"

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
            print("No data present, switching from auto sampler to grid sampler.")
            if policy_type == "auto": policy_type = "grid"


        #generate policies
        pols = [ [0.0] * pol_len for i in range(sample_count)]
        if not policies:
            #no policies were given, so generate them
            if policy_type == "auto":
                #ask the gp object for it's next guesses
                # HBayesGP.suggest_next(self,number_of_suggestions, minimum_climbs=40):
                print("Querying GP for new sample positions.")
                pols = self.GP.suggest_next(sample_count, minimum_climbs=10, CLIMB_FROM_CURRENT_DATA=False)
            elif policy_type == "grid":
                print("Getting grid coordinates.")
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
            new_class = []
            new_inbounds = []
            new_real = []

            print("Policies generated. Simulating...")
            for p in pols:

                #check filters
                pol_type = self.classify_policy(p)

                #check bounds
                pol_oob = self.check_if_OOB_policy(p)

                if pol_oob:
                    #it's an out-of-bounds policy
                    if not self.USING_OOB_PENALTY:
                        #just using the default OOB values
                        new_y.append(self.OOB_default_mean + 1 - 1)
                        new_y_var.append(self.OOB_default_var + 1 - 1)
                        new_X.append(p[:])
                        new_supprate.append(0.5) #TODO What should it be???
                        new_class.append(GOOD_POL)
                        new_inbounds.append(False)
                        new_real.append(False)

                    else:
                        #using the squared distance penalty
                        print("...a policy is OOB")
                        _y, _var, _X, _supp = self.SWM_sample(p)
                        penalty = self.penalize_OOB_sim_value(p)
                        print("sim val: {0}   penalty: {1}   policy: {2}".format(_y, penalty, p))
                        new_y.append(_y - penalty)
                        new_y_var.append(self.OOB_default_var + 1 - 1)
                        new_X.append(_X)
                        new_supprate.append(_supp)
                        new_class.append(self._supp_to_class(_supp))
                        new_inbounds.append(False)
                        new_real.append(True)


                elif pol_type == LB_POL:
                    #it's been classified as an LB policy, so use the current estimate of the 
                    #LB mean and variance, etc...
                    new_y.append(self.LB_default_mean + 1 - 1)
                    new_y_var.append(self.edge_default_var + 1 - 1)
                    new_X.append(p[:])
                    new_supprate.append(0.0)
                    new_class.append(LB_POL)
                    new_inbounds.append(True)
                    new_real.append(False)

                elif pol_type == SA_POL:
                    #it's been classified as an SA policy, so use the current estimate of the
                    # SA mean and variance
                    new_y.append(self.SA_default_mean + 1 - 1)
                    new_y_var.append(self.edge_default_var + 1 - 1)
                    new_X.append(p[:])
                    new_supprate.append(1.0)
                    new_class.append(SA_POL)
                    new_inbounds.append(True)
                    new_real.append(False)

                else:
                    #it's a good policy (GOOD_POL), so sample it
                    _y, _var, _X, _supp = self.SWM_sample(p)
                    new_y.append(_y)
                    new_y_var.append(_var)
                    new_X.append(_X)
                    new_supprate.append(_supp)
                    new_class.append(self._supp_to_class(_supp))
                    new_inbounds.append(True)
                    new_real.append(True)


            #add new data to the current data set
            self.y = self.y + new_y
            self.y_var = self.y_var + new_y_var
            self.X = self.X + new_X
            self.supprate = self.supprate + new_supprate
            self.classification = self.classification + new_class
            self.in_bounds = self.in_bounds + new_inbounds
            self.real_sample = self.real_sample + new_real

            print("Simulations complete. Fitting GP...")
            #if there's already a GP object, then add data to it
            if self.GP:
                self.GP.add_data(new_X, new_y, new_y_var, NO_FIT=True)
            else:
                #there's no GP, so make one with ALL of the current data (not just the new ones)
                self.GP = HBayesGP.HBayesGP(self.X, self.y, self.y_var, self.bounds)


            #re-fit the GP and classifier models
            self.fit_GP() # <-- if there's no GP, then this function will also add ALL the data to a new one.

            print("Fitting Classfier...")
            if self.USING_CLASSIFIER: self.fit_classifier()

            print("Additional Data Acquisition is Complete.")
            print("")

    #Uses existing data to fit the edge-policy classifier
    def fit_classifier(self):
        """Uses existing data to fit the edge-policy classifier.

        This function will skip any samples flagged as False in the self.real_sample list.
        Those samples are ones that have already been estimated by the classifier, and need
        to be excluded from the fit. Otherwise it'll be ever-more biased towards its earlier
        predictions, since those predictions would show up as (perfectly accurate) samples.

        PARAMETERS
        None

        RETURNS
        None 
        """

        #make sure the classifier has been initilaized
        if not self.classifier: self._init_classifier()

        #in other words, take self.X[i] if self.real_sample == True, and make a list out of it
        # and make that into a numpy array.
        filtered_X = np.array([ i for i,j in zip(self.X, self.real_sample) if j])
        filtered_class = np.array([ i for i,j in zip(self.classification, self.real_sample) if j])

        #fit the classifier
        self.classifier.fit( filtered_X, filtered_class )

    #Check's a policy against the current classifier system
    def classify_policy(self, policy):
        """ Runs the classifier (but not a bounds check), and returns an integer reflecting the policy type

        PARAMETERS
        policy: a policy of the same length as those used everywhere else

        RETURNS
        policy type: an integer set to equal one of the following constants:
            GOOD_POL, SA_POL, LB_POL
        """
        
        #policy is in-bounds; run the classfier if we're using one.
        if self.USING_CLASSIFIER:
            pol_type = self.classfier.predict([policy])
            #it returns a list of predictions from a list of policies
            # so just return the first element of that list, which should be the prediction
            return pol_type[0]  
        else:
            #it was in bounds, and we're not bothering with a classifier,
            #so return good_pol
            return GOOD_POL

    #Check's if a policy is in or out of boundss
    def check_if_OOB_policy(self, policy):
        """Returns true if policy is out of bounds"""
        pol_len = 2
        if self.USING_SWM2_1: pol_len = 6

        for i in range(pol_len):
            #      pol[i] < lower bound for i  or     pol[i] > upper bound for i
            if (policy[i] < self.bounds[i][0]) or (policy[i] > self.bounds[i][1]):
                return True

        #we've completed the loop without returning True, so this must be in bounds
        return False

    #Get the penalty associated with this policy for being OOB
    def penalize_OOB_sim_value(self, policy):
        """returns the out-of-bounds penalty for this policy"""
        pol_len = 2
        if self.USING_SWM2_1: pol_len = 6

        furthest = 0.0

        for i in range(pol_len):
            if (policy[i] < self.penalty_bounds[i][0]):
                furthest = max(furthest, (self.penalty_bounds[i][0] - policy[i]))
            elif (policy[i] > self.penalty_bounds[i][1]):
                furthest = max(furthest, (policy[i] - self.penalty_bounds[i][1]))

        return (self.OOB_alpha * furthest * furthest)

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
        plb = lb + self.default_penalty_distance
        pub = ub - self.default_penalty_distance
        if not new_bounds:
            if self.USING_SWM2_1:
                self.bounds = [[lb,ub],[lb,ub],[lb,ub],[lb,ub],[lb,ub],[lb,ub]]
                self.penalty_bounds = [[plb,pub],[plb,pub],[plb,pub],[plb,pub],[plb,pub],[plb,pub]]
            else:
                self.bounds = [[lb,ub],[lb,ub]]
                self.penalty_bounds = [[plb,pub],[plb,pub]]
        else:
            self.bounds = new_bounds[:]
            _p = self.default_penalty_distance
            self.penalty_bounds = [ [new_bounds[i][0] + _p, new_bounds[i][1] - _p] for i in len(new_bounds)]

    #report stats on current data set
    def report_all(self):
        """
        Reports:
        Dataset size
        Best Value
        Best Policy
        Number/Percent of LB, SA data points
        Number/Percent OOB data points
        Current Length Scale
        General Bounds
        LBFGS-B Bounds
        """

        #get counts
        all_sims = len(self.y)
        real_sims = len([i for i in self.real_sample if i])
        est_sims = all_sims - real_sims

        sa_sims = len([c for c, r in zip(self.classification, self.real_sample) if ((c == SA_POL) and (r))])
        lb_sims = len([c for c, r in zip(self.classification, self.real_sample) if ((c == LB_POL) and (r))])
        good_sims = len([c for c, r in zip(self.classification, self.real_sample) if ((c == GOOD_POL) and (r))])

        oob_sims = len([b for b, r in zip(self.in_bounds, self.real_sample) if ((not b) and (r))])

        print("Dataset Size, ALL:  {0}".format(all_sims))
        print("Dataset Size, Real: {0}  ({1}%)".format(real_sims, (round((100 * real_sims / all_sims),3))))
        print("Dataset Size, Est:  {0}  ({1}%)".format(est_sims, (round((100 * est_sims / all_sims),3))))
        print("")
        print("Actual SWM simulations per sample point: {0}".format(self.sims_per_sample))
        print("Total SWM simulations rum: {0}".format(self.sims_per_sample * real_sims))
        print("Highest Value Seen: {0}".format(max(self.y)))
        print("Highest Val. Policy: {0}".format(repr(self.GP.history_global_best_coords)))
        print("")
        print("Classifcations of Real Simulations (excluding estimated sims)")
        print("Good Simulations: {0}   ({1}%)".format(good_sims, (round((100 * good_sims / real_sims),3))))
        print("SA Simulations:   {0}   ({1}%)".format(sa_sims,   (round((100 * sa_sims / real_sims),3))))
        print("LB Simulations:   {0}   ({1}%)".format(lb_sims,   (round((100 * lb_sims / real_sims),3))))
        print("Total:            {0}".format(sa_sims + lb_sims + good_sims))
        print("")
        print("OOB Simulations:  {0}".format(oob_sims))
        print("")
        print("Most Recent Length Scale: {0}".format(self.GP.gp_length_scale))
        print("Bounds: {0}".format(repr(self.bounds)))

    #plot the first two dimensions of the GP
    def plot_gp(self, dim0=1, dim1=0):
        self.GP.plot_gp(self.bounds[0],self.bounds[1],50,0,dim0,dim1)

    #clear all data but leave other settings as-is
    def clear_data(self):

        ## Re-nitilaize classifier ##
        self.classfier = None
        self._init_classifier()
        
        ## DATA ##
        self.X = []
        self.y = []
        self.y_var = []
        self.supprate = []
        self.classification = []
        self.in_bounds = []
        self.real_sample = [] #True for an actual sample, False for one that's been predicted

        ## GAUSSIAN PROCESS OBJECT ##
        #this will be instantiated the first time that data is recieved
        self.GP = None

    ### PRIVATE FUNCTIONS ###

    #instantiates the classifier object(s)
    def _init_classifier(self):
        if self.classifier_type == "random forest":
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

        total_policy_count = 1
        for i in range(pol_len):
            total_policy_count *= slice_counts[i]

        if total_policy_count > 1000:
            print("WARNING: as currently parameterized, the grid sampler will visit {0} points".format(total_policy_count))
            x = raw_input("Do you want to continue? (y/n)")
            if not x == "y":
                return []


        #get slices
        slices = []
        for i in range(pol_len):
            slices.append(np.linspace(self.bounds[i][0], self.bounds[i][1], slice_counts[i]))


        #define recursive function
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

        #get coordinates as x and y vectors
        coordinate_list = []
        coordinate_list = _recur_get_next_line(slices, 0, [0.0] * pol_len, coordinate_list)


        return coordinate_list

    #gives the class, based on a suppression rate
    def _supp_to_class(self, supp_rate):
        if supp_rate >= 0.995: return SA_POL
        elif supp_rate <= 0.005: return LB_POL
        else: return GOOD_POL