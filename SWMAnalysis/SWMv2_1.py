"""SWM, A Simple Wildfire-inspired MDP model. Version 2.1"""


"""
SWMv2.1

New Features:

-An additional weather variable ("moisture") which works against the spread of large fires and which has
 a separate threshold.

-The policy function now includes the current state variables "timber" and "vulnerability" as parameters 
 in addition to the original. The full policy is now [CONS, HEAT, MOISTURE, TIMBER, VULNERABILITY, HABITAT]

-habitat value: a seperate reward structure with different criteria than the main "reward"

-Using feature transformations so that the logistic sees only features which have mean ~= 0 and STD ~= 0.5

-OPTIONALLY using feature vector length adjustment so that the crossproduct inside of the logistic policy 
 function is in the same range, regardless of how many values are in the vector
"""

import random, math, MDP
import numpy as np
from feature_transformation import feature_transformation as feature_trans

def simulate(timesteps, policy=[0,0,0,0,0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True):
    
    random.seed(random_seed)
    
    #range of the randomly drawn, uniformally distributed "event" that corresponds to "heat" and "moisture"
    event_max = 1.0
    event_min = 0.0

    #state variable bounds
    vuln_max = 1.0
    vuln_min = 0.02

    timber_max = 1.0
    timber_min = 0.0
    
    #timber multiplier: this is to scale the rewards while leaving the raw timber values between 0 and 1
    timber_multiplier = 10.0



    timesteps = int(timesteps)


    #sanitize policy
    pol = sanitize_policy(policy)


    #REWARD STRUCTURE
    
    #cost of suppression in a mild event
    supp_cost_mild = 9
    if "Suppression Cost - Mild Event" in model_parameters.keys(): supp_cost_mild = model_parameters["Suppression Cost - Mild Event"]

    #cost of suppresion in a severe event
    supp_cost_severe = 13
    if "Suppression Cost - Severe Event" in model_parameters.keys(): supp_cost_severe = model_parameters["Suppression Cost - Severe Event"]

    #cost of a severe fire on the next timestep
    burn_cost = 40
    if "Severe Burn Cost" in model_parameters.keys(): burn_cost = model_parameters["Severe Burn Cost"]


    #TRANSITION VARIABLES

    vuln_change_after_suppression = 0.01
    vuln_change_after_mild = -0.01
    vuln_change_after_severe = -0.015
    if "Vulnerability Change After Suppression" in model_parameters.keys(): vuln_change_after_suppression = model_parameters["Vulnerability Change After Suppression"]
    if "Vulnerability Change After Mild" in model_parameters.keys(): vuln_change_after_mild = model_parameters["Vulnerability Change After Mild"]
    if "Vulnerability Change After Severe" in model_parameters.keys(): vuln_change_after_severe = model_parameters["Vulnerability Change After Severe"]

    timber_change_after_suppression = 0.01
    timber_change_after_mild = 0.01
    timber_change_after_severe = -0.5
    if "Timber Value Change After Suppression" in model_parameters.keys(): timber_change_after_suppression = model_parameters["Timber Value Change After Suppression"]
    if "Timber Value Change After Mild" in model_parameters.keys(): timber_change_after_mild = model_parameters["Timber Value Change After Mild"]
    if "Timber Value Change After Severe" in model_parameters.keys(): timber_change_after_severe = model_parameters["Timber Value Change After Severe"]

    #habitat transition variables
    habitat_mild_maximum = 15
    habitat_mild_minimum = 0
    habitat_severe_maximum = 40
    habitat_severe_minimum = 10
    habitat_loss_if_no_mild = 0.2
    habitat_loss_if_no_severe = 0.2
    habitat_gain = 0.1


    if "Probabilistic Choices" in model_parameters.keys():
        if model_parameters["Probabilistic Choices"] == "True":
            PROBABILISTIC_CHOICES = True
        else:
            PROBABILISTIC_CHOICES = False



    #starting_condition = 0.8
    starting_Vulnerability = random.uniform(0.2,0.8)
    if "Starting Vulnerability" in model_parameters.keys(): starting_condition = model_parameters["Starting Condition"]
    starting_timber = random.uniform(0.2,0.8)
    if "Starting Timber Value" in model_parameters.keys(): starting_timber = model_parameters["Starting Timber Value"]
    starting_habitat = random.uniform(0.2,0.8)
    if "Starting Habitat Value" in model_parameters.keys(): starting_habitat = model_parameters["Starting Habitat Value"]

    #setting 'enums'
    MILD=0
    SEVERE=1


    #starting simulations
    states = [None] * timesteps

    #start current condition randomly among the three states
    current_vulnerability = starting_Vulnerability
    current_timber = starting_timber
    current_habitat = starting_habitat
    time_since_severe = 0
    time_since_mild = 0

    #instantiating the feature list now, to save cycles
    curent_features = [0.0, 0.0, 0.0, 0.0, 0.0]

    for i in range(timesteps):

        #generate the two weather variables for this event
        heat = random.uniform(event_min, event_max)
        moisture = random.uniform(event_min, event_max)

        #severity is meant to be a hidden, "black box" variable inside the MDP
        # and not available to the logistic function as a parameter
        severity = MILD
        if heat >= (1 - current_vulnerability): 
            #if moisture < ((event_max - event_min) * 0.3):
            if True:
                severity = SEVERE


        #logistic function for the policy choice
        curent_features = [heat, moisture, current_timber, current_vulnerability, current_habitat]
        policy_value = policy_function(curent_features, pol)

        #rolling a value to compare to the policy 'probability' in policy_value
        choice_roll = random.uniform(0,1)

        #assume let-burn for the moment...
        choice = False
        choice_prob = 1.0 - policy_value
        #check for suppress, and update values if necessary
        if PROBABILISTIC_CHOICES:
            if choice_roll < policy_value:
                choice = True
                choice_prob = policy_value
        else:
            if policy_value >= 0.5:
                choice = True
                choice_prob = policy_value


        ### CALCULATE PRIMARY REWARD ###
        supp_cost = 0
        burn_penalty = 0
        if choice:
            #suppression was chosen
            if severity == MILD:
                supp_cost = supp_cost_mild
            elif severity == SEVERE: 
                supp_cost = supp_cost_severe
        else:
            #suppress was NOT chosen
            if severity == SEVERE:
                #set this timestep's burn penalty to the value given in the overall model parameter
                #this is modeling the timber values lost in a large fire.
                burn_penalty = burn_cost

        current_reward = (timber_multiplier * current_timber) - supp_cost - burn_penalty
        #current_reward = 10 + (timber_multiplier * current_timber) - supp_cost - burn_penalty



        #Record state information
        #states[i] = [current_vulnerability, current_timber, heat, moisture, choice, choice_prob, policy_value, current_reward, current_habitat, i]
        states[i] = {
                      "Vulnerability": current_vulnerability,
                      "Timber": current_timber,
                      "Heat": heat,
                      "Moisture": moisture,
                      "Choice": choice,
                      "Choice Probability": choice_prob,
                      "Policy Value": policy_value,
                      "Reward": current_reward,
                      "Habitat": 10 * current_habitat,
                      "Time Step": i
                    }


        ### TRANSITION ###
        if not choice:
            #no suppression
            if severity == SEVERE:
                current_vulnerability += vuln_change_after_severe
                current_timber += timber_change_after_severe

                #reset both timers
                time_since_severe = 0
                time_since_mild += 1

            elif severity == MILD:
                current_vulnerability += vuln_change_after_mild
                current_timber += timber_change_after_mild

                #reset mild, increment severe
                time_since_mild = 0
                time_since_severe += 1
        else:
            #suppression
            current_vulnerability += vuln_change_after_suppression
            current_timber += timber_change_after_suppression

            #increment both timers
            time_since_mild += 1
            time_since_severe += 1

        #check for habitat changes. 
        #Note to self: suppression effects are already taken into account above
        if ( (time_since_mild <= habitat_mild_maximum) and 
             (time_since_mild >= habitat_mild_minimum) and
             (time_since_severe <= habitat_severe_maximum) and
             (time_since_severe >= habitat_severe_minimum)  ):

            #this fire is happy on all counts
            current_habitat += habitat_gain
        else:
            #this fire is unhappy in some way.
            if (time_since_mild > habitat_mild_maximum) or (time_since_mild < habitat_mild_minimum):
                current_habitat -= habitat_loss_if_no_mild
            if (time_since_severe > habitat_severe_maximum) or (time_since_severe < habitat_severe_minimum):
                current_habitat -= habitat_loss_if_no_severe



        #Enforce state variable bounds
        if current_vulnerability > vuln_max: current_vulnerability = vuln_max
        if current_vulnerability < vuln_min: current_vulnerability = vuln_min
        if current_timber > timber_max: current_timber = timber_max
        if current_timber < timber_min: current_timber = timber_min
        if current_habitat > 1: current_habitat = 1
        if current_habitat < 0: current_habitat = 0


        


    #finished simulations, report some values
    vals = []
    hab = []
    suppressions = 0.0
    joint_prob = 1.0
    prob_sum = 0.0

    #state information is stored as:
    # states[i] = {
    #                   "Vulnerability": current_vulnerability,
    #                   "Timber": current_timber,
    #                   "Heat": heat,
    #                   "Moisture": moisture,
    #                   "Choice": choice,
    #                   "Choice Probability": choice_prob,
    #                   "Policy Value": policy_value,
    #                   "Reward": current_reward,
    #                   "Habitat": current_habitat,
    #                   "Time Step", i
    #                 }
    for i in range(timesteps):
        if states[i]["Choice"]: suppressions += 1
        joint_prob *= states[i]["Choice Probability"]
        prob_sum += states[i]["Choice Probability"]
        vals.append(states[i]["Reward"])
        hab.append(states[i]["Habitat"])
    ave_prob = prob_sum / timesteps

    summary = {
                "Average State Value": round(np.mean(vals),3),
                "Total Pathway Value": round(np.sum(vals),3),
                "STD State Value": round(np.std(vals),1),
                "Average Habitat Value": round(np.mean(hab),1),
                "Suppressions": suppressions,
                "Suppression Rate": round((float(suppressions)/timesteps),2),
                "Joint Probability": joint_prob,
                "Average Probability": round(ave_prob, 3),
                "ID Number": random_seed,
                "Timesteps": timesteps,
                "Generation Policy": policy,
                "Version": "2.1",
                "Vulnerability Min": vuln_min,
                "Vulnerability Max": vuln_max,
                "Vulnerability Change After Suppression": vuln_change_after_suppression,
                "Vulnerability Change After Mild": vuln_change_after_mild,
                "Vulnerability Change After Severe": vuln_change_after_severe,
                "Timber Value Min": timber_min,
                "Timber Value Max": timber_max,
                "Timber Value Change After Suppression": timber_change_after_suppression,
                "Timber Value Change After Mild": timber_change_after_mild,
                "Timber Value Change After Severe": timber_change_after_severe,
                "Suppression Cost - Mild": supp_cost_mild,
                "Suppression Cost - Severe": supp_cost_severe,
                "Severe Burn Cost": burn_cost
              }

    if not SILENT:
        print("")
        print("Simulation Complete - Pathway " + str(random_seed))
        print("Average State Value: " + str(round(np.mean(vals),1)) + "   STD: " + str(round(np.std(vals),1)))
        print("Average Habitat Value: " + str(round(np.mean(hab),1)) )
        print("Suppressions: " + str(suppressions))
        print("Suppression Rate: " + str(round((float(suppressions)/timesteps),2)))
        print("Joint Probability:" + str(joint_prob))
        print("Average Probability: " + str(round(ave_prob, 3)))
        print("")


    summary["States"] = states

    return summary

def policy_function(feature_vector, policy, vector_length_scaling=False):
    """Calculates and returns the logistic policy function's value of a given feature vector"""

    #get the transformed values of the features
    transformed_features = feature_trans(feature_vector)

    #add the constant term
    transformed_features = [1.0] + transformed_features

    #do the raw crossproduct of the features times the values
    cross_product = np.sum(  [transformed_features[i] * policy[i] for i in range(len(policy))]  )

    #do the vector length scaling step, if desired
    if vector_length_scaling:
        cross_product = cross_product * (1.0 / np.sqrt(len(policy)))

    #get the logistic function value
    func_val = 1.0 / (1.0 + np.exp(-1.0 * cross_product))

    return func_val


def sanitize_policy(policy):
    pol = []
    if isinstance(policy, list):
        if len(policy) < 6:
            #it's under length for some reason, so append zeros
            z = [0] * (6-len(pol))
            pol = pol + z
        else:
            #the length is good, so just assign it. Extra values will just be ignored.
            pol = policy[:]
    else:
        #it's not a list, so find out what string it is
        if policy == 'LB':    pol = [-20,0,0,0,0,0]
        elif policy == 'SA':  pol = [ 20,0,0,0,0,0]
        elif policy == 'CT':  pol = [  0,0,0,0,0,0]
        else: pol = [0,0,0,0,0,0] #using CT as a catch-all for when the string is "MIXED_CT" or whatnot

    return pol

def convert_to_MDP_pathway(SWMv2_pathway,VALUE_ON_HABITAT=False, percentage_habitat=0):
    """ Converts a SWMv2 pathway into a generic MDP_Pathway object and returns it"""
    
    #create a new MDP pathway object, with policy length = 5
    new_MDP_pw = MDP.MDP_Pathway(6)
    
    new_MDP_pw.ID_number = SWMv2_pathway["ID Number"]
    new_MDP_pw.net_value = SWMv2_pathway["Total Pathway Value"]
    new_MDP_pw.actions_1_taken = SWMv2_pathway["Suppressions"]
    new_MDP_pw.actions_0_taken = SWMv2_pathway["Timesteps"] - SWMv2_pathway["Suppressions"]
    new_MDP_pw.generation_joint_prob = SWMv2_pathway["Joint Probability"]
    new_MDP_pw.set_generation_policy_parameters(SWMv2_pathway["Generation Policy"][:])
    
    for i in range(len(SWMv2_pathway["States"])):
        event = MDP.MDP_Event(i)

        #state information is stored as:
        # states[i] = {
        #                   "Vulnerability": current_vulnerability,
        #                   "Timber": current_timber,
        #                   "Heat": heat,
        #                   "Moisture": moisture,
        #                   "Choice": choice,
        #                   "Choice Probability": choice_prob,
        #                   "Policy Value": policy_value,
        #                   "Reward": current_reward,
        #                   "Habitat": current_habitat,
        #                   "Time Step", i
        #                 }
        
        heat = SWMv2_pathway["States"][i]["Heat"]
        moisture = SWMv2_pathway["States"][i]["Moisture"]
        timber = SWMv2_pathway["States"][i]["Timber"]
        vulnerabilty = SWMv2_pathway["States"][i]["Vulnerability"]
        habitat = SWMv2_pathway["States"][i]["Habitat"]
        event.state = [1, heat, moisture, timber, vulnerabilty, habitat ]

        event.state_length = 6
        event.action = SWMv2_pathway["States"][i]["Choice"]
        event.decision_prob = SWMv2_pathway["States"][i]["Choice Probability"]
        event.action_prob = SWMv2_pathway["States"][i]["Policy Value"]

        #setting value.  Value_on_habitat takes precedence, followed by percentage_habitat.
        #if neither are set, then the default is to use the regular (timber val - supp cost) reward.
        if VALUE_ON_HABITAT:
            event.rewards = SWMv2_pathway["States"][i]["Habitat"]
        elif percentage_habitat > 0:
            part_hab = SWMv2_pathway["States"][i]["Habitat"] * percentage_habitat
            part_val = SWMv2_pathway["States"][i]["Habitat"] * (1 - percentage_habitat)
            event.rewards = part_hab + part_val
        else:
            event.rewards = SWMv2_pathway["States"][i]["Reward"]
        
        new_MDP_pw.events.append(event)

    #everything needed for the MDP object has been filled in, so now
    # remove the states (at least) and add the rest of the SWM dictionary's entries as metadata
    SWMv2_pathway.pop("States",None)
    new_MDP_pw.metadata=SWMv2_pathway
    
    return new_MDP_pw
    
