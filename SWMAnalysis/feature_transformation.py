#SWM v2 feature transformation

import numpy as np
#import matplotlib.pyplot as plt


def feature_transformation(feature_vector):
    """Adjusts and rescales the values of SWMv2 features to fall within an appropriate range.

    ARGUEMENTS
    feature_vector: a list of the features values of a single state of the SWM v2 simulator

    RETURNS
    adjusted_vector: a list of the same length as feature_vector, containing the transformed
      values of each feature.
    
    """

    #create the return list
    adjusted_vector = [0.0] * len(feature_vector)

    #means and standard deviations from emperical data

    #values from get_means_and_stds()
    # heat ave: 0.500375296248
    # humidity ave: 0.500926956231
    # timber ave: 2.77865692101
    # vulnerability ave: 0.364513749217
    # habitat ave: 2.02501996769

    # heat STD: 0.288972166094
    # humidity STD: 0.288167851971
    # timber STD: 2.79279655522
    # vulnerability STD: 0.376634871739
    # habitat STD: 5.00511964501



    #                 HEAT      HUMID    TIMB     VULN    HAB
    feature_means = ( 0.5,      0.5,     2.7787,  0.3645, 2.0250)
    feature_STDs  = ( 0.2888,   0.2888,  2.7927,  0.3766, 5.0051)


    #the feature transformations are done one-at-a-time to allow for custom handling of each one.
    #The generic goal is:    mean=0    STD = 0.5

    #NOTE: the constant has not been added to the features, so in SWMv2.1, there are only 5 feature values

    #Transform feature 0 
    # "heat" value
    adjusted_vector[0] =  (feature_vector[0] - feature_means[0]) / (feature_STDs[0] * 2)

    #Transform feature 1
     # "humidity" value
    adjusted_vector[1] =  (feature_vector[1] - feature_means[1]) / (feature_STDs[1] * 2)

    #Transform feature 3
    # "timber" value
    adjusted_vector[2] =  (feature_vector[2] - feature_means[2]) / (feature_STDs[2] * 2)

    #Transform feature 4
    # "vulnerability" value
    adjusted_vector[3] =  (feature_vector[3] - feature_means[3]) / (feature_STDs[3] * 2)

    #Transform feature 5
    # "habitat" value
    adjusted_vector[4] =  (feature_vector[4] - feature_means[4]) / (feature_STDs[4] * 2)


    #return the transformed features
    return adjusted_vector



def heat_transformation(heat_value):
    """Get the transformed heat value (esp for SWMv1)"""
    input_vector = [heat_value, 0, 0, 0, 0]
    trans_vector = feature_transformation(input_vector)
    return trans_vector[0]