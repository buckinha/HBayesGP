

from SWMAnalysis import SWMAnalyst as SWMA

swma = SWMA.SWMAnalyst()
swma.set_SWM_model(2)
swma.grid_sampling_density = 25
swma.add_samples()

############
#Get samples
############

#get a round number of samples into swma


#get the sample matrix X, and y, from swma
#get the sample varainces, y_var from swma
X = swma.X[:]
y = swma.y[:]
y_var = swma.y_var[:]



#############################
#compute covariance matrix
#############################

#get the defnition for the covariance function
def cov_func(x1,x2):
    #ensure x1==x2 is handled
    return 0

#make the covarance matrix

#fill the cov.matrix using the covariance function

#lazy implementation... should instead do the top triangle and reflect
K = [ [cov_func(X[i],X[j]) for j in range(len(X)) ] for i in range(len(X)) ]

#input the nugget
nug_mult = 2.0
for i in range(len(X)):
    K[i][i] = y_var[i] * nug_mult

K = np.asmatrix(K)
Kinv = K.inv()


#################################
#draw a new point
################################
y_star = swma.GP._rand_coord()


##############################
#compute mean
#################################

#compute the covariance between y_star and each X
k_star = [ cov_func(y_star, _x) for _x in X]
k_T = np.transpose(k_star)

f_star = np.dot(np.dot(k_T, Kinv), y)


############################
#compute variance
##############################
kxx = cov_func(y_star, y_star)

Vf_star = kxx - np.dot(np.dot(k_T, Kinv), k_star)