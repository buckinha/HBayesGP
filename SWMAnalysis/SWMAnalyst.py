import SWMv1_3 as SWM1
import SWMv2_1 as SWM2
import HBayesGP, R_plotting
from neighbor_dist import neighbor_distances
import random, datetime, Queue
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
        #how many LB or SA sims before the rest are skipped. Set to -1 to ignore this parameter
        self.assume_policy_after = 5
        #using the larger model?
        self.USING_SWM2_1 = False
        self.SWM2_dimensions = 6

        #flag for whether to be grabbing the discounted or average stae value
        self.USING_DISCOUNTING = False

        #flag to indicade that data has changed and neighbor distances need to be re-computed
        self.RECALC_NEIGHBOR_DISTANCES = True
        self.all_distances = {}

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
        self.USING_CLASSIFIER = True
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
        self.classifier = None
        self._init_classifier()
        

        ## DATA ##
        self.X = []
        self.y = []
        self.y_var = []
        self.supprate = []
        self.classification = []
        self.in_bounds = [] #True for a sample that is in-bounds
        self.real_sample = [] #True for an actual sample, False for one that's been predicted
        self.SA_values = [] #individual simulation values from sims which were SA
        self.LB_values = [] #individual simulation values from sims which were LB

        ## BOUNDARIES ##
        self.bounds = []
        self.default_upper_bound_value =  25.0
        self.default_lower_bound_value = -25.0
        self.default_penalty_distance = 2.0
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
            vector with length = 2.  For SWMv2.1, this should be a length 6 (or longer?) vector


        RETURNS
        -------
        y: the average value of these simulations

        var: the variance of the values of these simulations

        X: the policy gven

        supprate: the average suppression rate of these simulations

        """

        
        sims = []
        simulator = SWM1
        if self.USING_SWM2_1: simulator = SWM2

        #SWMv2_1.simulate() function signature is:
        #def simulate(timesteps, policy=[0,0,0,0,0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True):
        if (self.assume_policy_after < 1) or (self.assume_policy_after >= self.sims_per_sample):
            #not ignoring any sims, even if the first several are all SA or all LB
            sims = [simulator.simulate(200, policy, random.random(), SILENT=True) for i in range(self.sims_per_sample)]
        else:
            #if the first 'n' simulations are uniformally SA or LB, we'll ignore the rest
            _LB_count = 0
            _SA_count = 0
            #get the first simulations
            sims1 = [simulator.simulate(200, policy, random.random(), SILENT=True) for i in range(self.assume_policy_after)]

            #check for SA and LB
            for s in sims1:
                if s["Suppression Rate"] > 0.995: _SA_count += 1
                if s["Suppression Rate"] < 0.005: _LB_count += 1

            if _SA_count >= self.assume_policy_after:
                #this is going to be assumed as an SA policy, so add these values to the list of SA sim values
                if self.USING_DISCOUNTING:
                    _vals = [ s["Discounted Value"] for s in sims1 ]
                    self.SA_values = self.SA_values + _vals
                else:
                    _vals = [ s["Average State Value"] for s in sims1 ]
                    self.SA_values = self.SA_values + _vals
                y = np.mean(self.SA_values)
                var = np.var(self.SA_values)
                return y, var, policy, 1.0

            elif _LB_count >= self.assume_policy_after:
                #this is going to be assumed as an LB policy...
                if self.USING_DISCOUNTING:
                    _vals = [ s["Discounted Value"] for s in sims1 ]
                    self.LB_values = self.LB_values + _vals
                else:
                    _vals = [ s["Average State Value"] for s in sims1 ]
                    self.LB_values = self.LB_values + _vals
                y = np.mean(self.LB_values)
                var = np.var(self.LB_values)
                return y, var, policy, 0.0

            else:
                #the first "n" sims didn't qualify as all SA or all LB, so simulate the rest and continue
                sims2 = [simulator.simulate(200, policy, random.random(), SILENT=True) for i in range(self.sims_per_sample - self.assume_policy_after)]
                sims = sims1 + sims2

        data = []
        if self.USING_DISCOUNTING: 
            data = [  sims[i]["Discounted Value"]  for i in range(self.sims_per_sample) ] 
        else:
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

        #flag for re-calculation of neighbor distances, since it's likely that the data will change
        self.RECALC_NEIGHBOR_DISTANCES = True

        pol_len = 2
        if self.USING_SWM2_1: pol_len = self.SWM2_dimensions


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
                pols = self.GP.suggest_next(sample_count, minimum_climbs=50, CLIMB_FROM_CURRENT_DATA=False)
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
                pol_type = GOOD_POL
                if len(self.y) > 0:
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
                        #print("...a policy is OOB")
                        _y, _var, _X, _supp = self.SWM_sample(p)
                        penalty = self.penalize_OOB_sim_value(p)
                        #print("sim val: {0}   penalty: {1}   policy: {2}".format(_y, penalty, p))
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
                    if len(self.LB_values) > 0:
                        new_y.append(np.mean(self.LB_values))
                        new_y_var.append(np.var(self.LB_values))
                    else:
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
                    if len(self.SA_values) > 0:
                        new_y.append(np.mean(self.SA_values))
                        new_y_var.append(np.var(self.SA_values))
                    else:
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

            print("Fitting Classifier...")
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
        if not hasattr(self, 'classifier'): self._init_classifier()

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
        
        #policy is in-bounds; run the classifier if we're using one.
        if self.USING_CLASSIFIER:
            pol_type = self.classifier.predict([policy])
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
        if self.USING_SWM2_1: pol_len = self.SWM2_dimensions

        if self.USING_OOB_PENALTY:
            for i in range(pol_len):
                #      pol[i] < lower bound for i  or     pol[i] > upper bound for i
                if (policy[i] < self.penalty_bounds[i][0]) or (policy[i] > self.penalty_bounds[i][1]):
                    return True
        else:
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
        if self.USING_SWM2_1: pol_len = self.SWM2_dimensions

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

    #sets bounds to given, or to default values if no args are given
    def set_bounds(self, new_bounds=None):
        if not new_bounds:
            lb = self.default_lower_bound_value
            ub = self.default_upper_bound_value
            plb = lb + self.default_penalty_distance
            pub = ub - self.default_penalty_distance
            if self.USING_SWM2_1:
                #TODO make this robust to varying policy lengths
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

        #calculate the number of simulations that have been run, all told
        total_sims_run = 0
        if self.assume_policy_after > 0:
            #some samples may only have been the result of a few simulations
            assumption_sims = len(self.SA_values) + len(self.LB_values)
            full_sample_sims = real_sims - assumption_sims
            total_sims_run = (self.sims_per_sample * full_sample_sims) + (self.assume_policy_after * assumption_sims)
        else:
            #not using that feature
            total_sims_run = self.sims_per_sample * real_sims

        sa_sims = len([c for c, r in zip(self.classification, self.real_sample) if ((c == SA_POL) and (r))])
        lb_sims = len([c for c, r in zip(self.classification, self.real_sample) if ((c == LB_POL) and (r))])
        good_sims = len([c for c, r in zip(self.classification, self.real_sample) if ((c == GOOD_POL) and (r))])

        oob_sims = len([b for b, r in zip(self.in_bounds, self.real_sample) if ((not b) and (r))])

        print("Dataset Size, ALL:  {0}".format(all_sims))
        print("Dataset Size, Real: {0}  ({1}%)".format(real_sims, (round((100 * real_sims / all_sims),3))))
        print("Dataset Size, Est:  {0}  ({1}%)".format(est_sims, (round((100 * est_sims / all_sims),3))))
        print("")
        print("Actual SWM simulations per sample point: {0}".format(self.sims_per_sample))
        print("Total SWM simulations run: {0}".format(total_sims_run))
        print("Highest Value Seen: {0}".format(max(self.y)))
        print("Highest Val. Policy: {0}".format(repr(self.GP.current_global_best_coord)))
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
        distances = self.get_all_neighbor_distances()
        print("")
        print("Average Dist to Neighbors:         " + str(distances["Average Distance"]))
        print("Average Dist to Furthest Neighbor: " + str(distances["Average Furthest"]))
        print("Average Dist to Nearest Neighbor:  " + str(distances["Average Nearest"]))
        print("Furthest two points: " + str(distances["Absolute Furthest"] ))
        print("Closest two points: " + str(distances["Absolute Nearest"]))
        print("Distance to the closest, 'furthest neighbor': " + str(distances["Shortest Furthest"]))
        print("Distance to the furthest, 'closest neighbor': " + str(distances["Furthest Nearest"]))


    #how many sims have been run all together (including with assumptions, etc...)?
    def get_sim_count(self):
        #get counts
        total_sims_run = 0
        real_sims = len([i for i in self.real_sample if i])
        if self.assume_policy_after > 0:
            #some samples may only have been the result of a few simulations
            assumption_sims = len(self.SA_values) + len(self.LB_values)
            full_sample_sims = real_sims - assumption_sims
            total_sims_run = (self.sims_per_sample * full_sample_sims) + (self.assume_policy_after * assumption_sims)
        else:
            #not using that feature
            total_sims_run = self.sims_per_sample * real_sims

        return total_sims_run

    #calculate all neighbor distances
    def get_all_neighbor_distances(self):
        if self.RECALC_NEIGHBOR_DISTANCES:
            self.all_distances = neighbor_distances(self.X)
            self.RECALC_NEIGHBOR_DISTANCES = False

        return self.all_distances

    #over all the points, what's the average dist to nearest neighbor?
    def get_ave_nearest_neighbor_dist(self):
        distances = self.get_all_neighbor_distances()
        return distances["Average Nearest"]

    #plot the first two dimensions of the GP
    def plot_gp(self, dim0=1, dim1=0):
        self.GP.plot_gp(self.bounds[0],self.bounds[1],50,0,dim0,dim1)

    #clear all data but leave other settings as-is
    def clear_data(self):

        ## Re-nitilaize classifier ##
        self.classifier = None
        self._init_classifier()
        
        ## DATA ##
        self.X = []
        self.y = []
        self.y_var = []
        self.supprate = []
        self.classification = []
        self.in_bounds = []
        self.real_sample = [] 
        self.SA_values = []
        self.LB_values = []

        ## GAUSSIAN PROCESS OBJECT ##
        self.GP = None


    #create R file to plot the Monte Carlo value surface
    def output_MC_surfaces(self, dim0=0, dim1=1, default_policy=[0,0,0,0,0,0], dim0_step=0.5, dim1_step=0.5, VALUE_ON_HABITAT=False):
        """Step through the SWM policy space and get the monte carlo net values at each policy point


        PARAMETERS
        dim0: the integer index value of the policy parameter on the graph's horizontal axis
        dim1: the integer index value of the policy parameter on the graph's vertical axis

        default_policy: default=[0,0,0,0,0,0]  The values to assign to each ppolicy parameter.
            The values for dim0 and dim1 are ignored.

        dim0_step: the approximate step size for incrementing dimension 0 across it's range
        dim1_step: the approcimate step size for incrementing dimension 1 accoss it's range

        VALUE_ON_HABITAT: Boolean flag indicating whether SWM2 simulations should be run with 
            the habitat quality as the recorded simulation value. Default is False, indicating 
            that the sim values are of costs/income units


        RETURNS
        None, but outputs the file "mc_value_graph.txt"

        """
        #checks for valid indices
        _indices_valid = True
        if self.USING_SWM2_1:
            if ((dim0 < 0) or (dim0>self.SWM2_dimensions-1) or (dim1 <0) or (dim1>self.SWM2_dimensions-1)):
                _indices_valid = False
        else:
            if ((dim0 < 0) or (dim0>1) or (dim1 <0) or (dim1>1)):
                _indices_valid = False
        if not _indices_valid:
            print("Invalid indices for the current SWM model: dim0:{0}  dim1:{1}".format(dim0,dim1))
            return None

        timesteps=200

        dim0_range=self.bounds[dim0][:]
        dim1_range=self.bounds[dim1][:]

        start_time = "Started:  " + str(datetime.datetime.now())

        #get step counts and starting points
        dim0_step_count = int(  abs(dim0_range[1] - dim0_range[0]) / dim0_step  ) + 1
        dim1_step_count = int(  abs(dim1_range[1] - dim1_range[0]) / dim1_step  ) + 1

        #flipping them if they're in reverse o
        dim0_start = dim0_range[0]
        if dim0_range[1] < dim0_range[0]: dim0_start = dim0_range[1]

        dim1_start = dim1_range[0]
        if dim1_range[1] < dim1_range[0]: dim1_start = dim1_range[1]

        dim0_vector = [dim0_start + (i*dim0_step) for i in range(dim0_step_count)]
        dim1_vector = [dim1_start + (i*dim1_step) for i in range(dim1_step_count)]

        #create the rows/columns structure
        val_cols = [ [0.0] * dim1_step_count for i in range(dim0_step_count) ]
        var_cols = [ [0.0] * dim1_step_count for i in range(dim0_step_count) ]
        supp_cols = [ [0.0] * dim1_step_count for i in range(dim0_step_count) ]


        #step through the polcies and generate monte carlo rollouts, and save their average values
        #i could use the dim0_vector and dim1_vectors here, but I need the row/column indices for
        # when I put the values into val_rows, etc...
        for row in range(dim1_step_count):
            for col in range(dim0_step_count):
                dim0_val = dim0_start + col*dim0_step
                dim1_val = dim1_start + row*dim1_step
              
                _cur_pol = default_policy[:]
                _cur_pol[dim0] = dim0_val
                _cur_pol[dim1] = dim1_val
                #self.SWM_sample(policy) returns: y, var, policy, supprate
                _y, _var, _pol, _supp = self.SWM_sample(_cur_pol)

                val_cols[col][row] = _y
                var_cols[col][row]= _var
                supp_cols[col][row] = _supp


        end_time = "Finished: " + str(datetime.datetime.now())

        #finished gathering output strings, now write them to the file
        f = open('mc_value_graph.txt', 'w')
        f_var = open('mc_variance_graph.txt', 'w')
        f_supp = open('mc_supp_rate_graph.txt', 'w')
        f_all = [f, f_var, f_supp]

        #Writing Header
        header = ""
        if self.USING_SWM2_1:
            header = "SWMAnalyst Output of Monte Carlo sims on SWM v2\n"
        else: 
            header = "SWMAnalyst Output of Monte Carlo sims on SWM v1\n"
        f.write(header)
        f_var.write(header)
        f_supp.write(header)
        if VALUE_ON_HABITAT:
            f.write("Values are of the Habitat Value index\n")
        else:
            f.write("Values are of Receipts - Costs\n")
        f_var.write("Values are the variance of the simulations at each point.\n")
        f_supp.write("Values are the average suppresion rate of the simulations at each point.\n")

        for _f in f_all:
            _f.write(start_time + "\n")
            _f.write(end_time + "\n")
            _f.write("Pathways per Point: " + str(self.sims_per_sample) +"\n")
            _f.write("Timesteps per Pathway: " + str(timesteps) +"\n")
            _f.write("dim0 Range: " + str(dim0_range) +"\n")
            _f.write("dim1 Range: " + str(dim1_range) +"\n")
            _f.write("default policy:" + repr(default_policy) + "\n")
            _f.write("\n")
            _f.write("dim0 policy values:" + repr(dim1_vector) + "\n")
            _f.write("dim1 policy values:" + repr(dim1_vector) + "\n")
            _f.write("\n")
            _f.write("Values:\n")

        
        for col in range(dim1_step_count):
            for row in range(dim0_step_count):
                f.write(     str(val_cols[col][row]))
                f_var.write( str(var_cols[col][row]))
                f_supp.write(str(supp_cols[col][row]))
                if row < (dim0_step_count - 1):
                    f.write(" ")
                    f_var.write(" ")
                    f_supp.write(" ")
            f.write("\n")
            f_var.write("\n")
            f_supp.write("\n")

        f_all = []
        f.close()
        f_var.close()
        f_supp.close()

        #now create the actual R plots, since the data has been written
        #figure out which titles and labels to use
        xlab=""
        ylab=""
        title=""
        if dim1 == 0:
            title = title + "Constant vs "
            ylab = "Policy Constant"
        if dim1 == 1:
            title = title + "Weather vs "
            ylab = "Policy Parameter on Weather Severity"
        if dim1 == 2:
            title = title + "Moisture vs "
            ylab = "Policy Parameter on Moisture"
        if dim1 == 3:
            title = title + "Timber vs "
            ylab = "Policy Parameter on Timber"
        if dim1 == 4:
            title = title + "Vulnerability vs "
            ylab = "Policy Parameter on Vulnerability"
        if dim1 == 5:
            title = title + "Habitat vs "
            ylab = "Policy Parameter on Habitat"

        
        if dim0 == 0:
            title = title + "Constant"
            xlab = "Policy Constant"
        if dim0 == 1:
            title = title + "Weather"
            xlab = "Policy Parameter on Weather Severity"
        if dim0 == 2:
            title = title + "Moisture"
            xlab = "Policy Parameter on Moisture"
        if dim0 == 3:
            title = title + "Timber"
            xlab = "Policy Parameter on Timber"
        if dim0 == 4:
            title = title + "Vulnerability"
            xlab = "Policy Parameter on Vulnerability"
        if dim0 == 5:
            title = title + "Habitat"
            xlab = "Policy Parameter on Habitat"

        R_plotting.create_R_plots(dim0_vector, dim1_vector, title, xlab, ylab)
        print("Process Complete. 10 files written.")




    ### PRIVATE FUNCTIONS ###

    #instantiates the classifier object(s)
    def _init_classifier(self):
        if self.classifier_type == "random forest":
            self.classifier = RandomForestClassifier(max_depth=self.RFC_depth, n_estimators=self.RFC_estimators)
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
        if self.USING_SWM2_1: pol_len = self.SWM2_dimensions

        slice_counts = [0] * pol_len
        for i in range(pol_len):
            slice_counts[i] = abs(self.bounds[i][0] - self.bounds[i][1]) / self.grid_sampling_density
            slice_counts[i] = np.ceil(slice_counts[i])
            #catch the case when bounds have been restricted to something like [0,0] or [10,10]
            if slice_counts[i] == 0: slice_counts[i] = 1

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