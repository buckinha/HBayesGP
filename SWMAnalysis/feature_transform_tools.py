import SWMv2_1 as SWM2
import numpy as np
from feature_transformation import feature_transformation

def get_means_and_stds(years=200, ct_count=100, lb_count=100, sa_count=100, opt2d_count=100, _return_for_test=False):


    #create the pathways

    pw_ct = [SWM2.simulate(years,'CT', random_seed=i+1000, SILENT=True) for i in range(ct_count)]
    pw_lb = [SWM2.simulate(years,'LB', random_seed=i+2000,  SILENT=True) for i in range(lb_count)]
    pw_sa = [SWM2.simulate(years,'SA', random_seed=i+3000,  SILENT=True) for i in range(sa_count)]
    pw_opt = [SWM2.simulate(years,[-16,20,0,0,0,0], random_seed=i+4000,  SILENT=True) for i in range(opt2d_count)]


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

    #features = [[1.0,all_vals[0][i],all_vals[1][i],all_vals[2][i],all_vals[3][i],all_vals[4][i]] for i in range(len(all_vals[0]))]
    features = [[all_vals[0][i],all_vals[1][i],all_vals[2][i],all_vals[3][i],all_vals[4][i]] for i in range(len(all_vals[0]))]

    trans_features = [feature_transformation(features[i]) for i in range(len(features))]

    trans_heat  = [trans_features[i][0] for i in range(len(features))]
    trans_humid = [trans_features[i][1] for i in range(len(features))] 
    trans_timb  = [trans_features[i][2] for i in range(len(features))] 
    trans_vuln  = [trans_features[i][3] for i in range(len(features))] 
    trans_hab   = [trans_features[i][4] for i in range(len(features))] 

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
