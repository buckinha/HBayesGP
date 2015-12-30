#SWM v2 feature transformation

import SWMv2_1 as SWM
#import SWMv2_Trials as SWMT
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
    # timber ave: 0.477938331883
    # vulnerability ave: 0.374390081586
    # habitat ave: 1.57126067857

    # heat STD: 0.288972166094
    # humidity STD: 0.288167851971
    # timber STD: 0.412139275761
    # vulnerability STD: 0.372340080948
    # habitat STD: 3.28672846053


    #                 HEAT      HUMID    TIMB     VULN    HAB
    feature_means = ( 0.5,      0.5,     0.4779,  0.3744, 1.5713)
    feature_STDs  = ( 0.2888,   0.2888,  0.4121,  0.3723, 3.2867)


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



def get_means_and_stds(years=200, ct_count=100, lb_count=100, sa_count=100, opt2d_count=100, _return_for_test=False):


    #create the pathways

    pw_ct = [SWM.simulate(years,'CT', random_seed=i+1000, SILENT=True) for i in range(ct_count)]
    pw_lb = [SWM.simulate(years,'LB', random_seed=i+2000,  SILENT=True) for i in range(lb_count)]
    pw_sa = [SWM.simulate(years,'SA', random_seed=i+3000,  SILENT=True) for i in range(sa_count)]
    pw_opt = [SWM.simulate(years,[-16,20,0,0,0,0], random_seed=i+4000,  SILENT=True) for i in range(opt2d_count)]


    #tabulate values
    ct_heat = []
    ct_moisture = []
    ct_timber = []
    ct_vuln = []
    ct_hab = []

    lb_heat = []
    lb_moisture = []
    lb_timber = []
    lb_vuln = []
    lb_hab = []

    sa_heat = []
    sa_moisture = []
    sa_timber = []
    sa_vuln = []
    sa_hab = []

    opt_heat = []
    opt_moisture = []
    opt_timber = []
    opt_vuln = []
    opt_hab = []


    for pw in pw_ct:
      ct_heat   = ct_heat   + [pw["States"][j]["Heat"]          for j in range(years)]
      ct_moisture  = ct_moisture  + [pw["States"][j]["Moisture"]      for j in range(years)]
      ct_timber = ct_timber + [pw["States"][j]["Timber"]        for j in range(years)]
      ct_vuln   = ct_vuln   + [pw["States"][j]["Vulnerability"] for j in range(years)]
      ct_hab    = ct_hab    + [pw["States"][j]["Habitat"]       for j in range(years)]

    for pw in pw_lb:
      lb_heat   = lb_heat   + [pw["States"][j]["Heat"]          for j in range(years)]
      lb_moisture  = lb_moisture  + [pw["States"][j]["Moisture"]      for j in range(years)]
      lb_timber = lb_timber + [pw["States"][j]["Timber"]        for j in range(years)]
      lb_vuln   = lb_vuln   + [pw["States"][j]["Vulnerability"] for j in range(years)]
      lb_hab    = lb_hab    + [pw["States"][j]["Habitat"]       for j in range(years)]

    for pw in pw_sa:
      sa_heat   = sa_heat   + [pw["States"][j]["Heat"]          for j in range(years)]
      sa_moisture  = sa_moisture  + [pw["States"][j]["Moisture"]      for j in range(years)]
      sa_timber = sa_timber + [pw["States"][j]["Timber"]        for j in range(years)]
      sa_vuln   = sa_vuln   + [pw["States"][j]["Vulnerability"] for j in range(years)]
      sa_hab    = sa_hab    + [pw["States"][j]["Habitat"]       for j in range(years)]

    for pw in pw_opt:
      opt_heat   = opt_heat   + [pw["States"][j]["Heat"]          for j in range(years)]
      opt_moisture  = opt_moisture  + [pw["States"][j]["Moisture"]      for j in range(years)]
      opt_timber = opt_timber + [pw["States"][j]["Timber"]        for j in range(years)]
      opt_vuln   = opt_vuln   + [pw["States"][j]["Vulnerability"] for j in range(years)]
      opt_hab    = opt_hab    + [pw["States"][j]["Habitat"]       for j in range(years)]


    #combine values by concatenating into single lists

    all_heat     = ct_heat     + lb_heat     + sa_heat     + opt_heat
    all_humidity = ct_moisture + lb_moisture + sa_moisture + opt_moisture
    all_timber   = ct_timber   + lb_timber   + sa_timber   + opt_timber
    all_vuln     = ct_vuln     + lb_vuln     + sa_vuln     + opt_vuln
    all_hab      = ct_hab      + lb_hab      + sa_hab      + opt_hab


    #print values
    if not _return_for_test:
        print("heat ave: " + str(np.mean(all_heat)))
        print("humidity ave: " + str(np.mean(all_humidity)))
        print("timber ave: " + str(np.mean(all_timber)))
        print("vulnerability ave: " + str(np.mean(all_vuln)))
        print("habitat ave: " + str(np.mean(all_hab)))
        print("")
        print("heat STD: " + str(np.std(all_heat)))
        print("humidity STD: " + str(np.std(all_humidity)))
        print("timber STD: " + str(np.std(all_timber)))
        print("vulnerability STD: " + str(np.std(all_vuln)))
        print("habitat STD: " + str(np.std(all_hab)))


    if _return_for_test:
        return [all_heat, all_humidity, all_timber, all_vuln, all_hab]
    else:
        return "...Process Complete"


def test_feature_means():
    all_vals = get_means_and_stds(_return_for_test=True)

    features = [[1.0,all_vals[0][i],all_vals[1][i],all_vals[2][i],all_vals[3][i],all_vals[4][i]] for i in range(len(all_vals[0]))]

    trans_features = [feature_transformation(features[i]) for i in range(len(features))]

    trans_heat  = [trans_features[i][1] for i in range(len(features))]
    trans_humid = [trans_features[i][2] for i in range(len(features))] 
    trans_timb  = [trans_features[i][3] for i in range(len(features))] 
    trans_vuln  = [trans_features[i][4] for i in range(len(features))] 
    trans_hab   = [trans_features[i][5] for i in range(len(features))] 

    #print values
    print("Means and STDs of transformed features:")
    print("")
    print("heat ave: " + str(np.mean(trans_heat)))
    print("humidity ave: " + str(np.mean(trans_humid)))
    print("timber ave: " + str(np.mean(trans_timb)))
    print("vulnerability ave: " + str(np.mean(trans_vuln)))
    print("habitat ave: " + str(np.mean(trans_hab)))
    print("")
    print("heat STD: " + str(np.std(trans_heat)))
    print("humidity STD: " + str(np.std(trans_humid)))
    print("timber STD: " + str(np.std(trans_timb)))
    print("vulnerability STD: " + str(np.std(trans_vuln)))
    print("habitat STD: " + str(np.std(trans_hab)))


    #VERIFIED OUTPUT:   Means ~= 0,  STDs ~= 0.5
    # heat ave: 0.000649751122486
    # humidity ave: 0.00160484112048
    # timber ave: 4.65079875667e-05
    # vulnerability ave: -1.33204590571e-05
    # habitat ave: -5.9819018408e-06

    # heat STD: 0.500298071493
    # humidity STD: 0.498905560891
    # timber STD: 0.500047653192
    # vulnerability STD: 0.500053828831
    # habitat STD: 0.500004329652
