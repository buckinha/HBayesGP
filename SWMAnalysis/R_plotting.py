#R-plotting for SWMAnalyst

def write_R_script_for_MC_surface_maps(x_vector, y_vector, title="", x_label="", y_label="", OUTPUT_FILES_ONLY=False):
    
    f = open('mc_surfaces_R_script.R','w')
    f.write("#This script will plot the associated surface maps output by SWMAnalyst.\n")
    f.write("\n")

    f.write("#Read in the files\n")
    f.write("z_val <- as.matrix(read.table(\"mc_value_graph.txt\", header=False, sep=\" \", skip=14))\n")
    f.write("z_var <- as.matrix(read.table(\"mc_variance_graph.txt\", header=False, sep=\" \", skip=14))\n")
    f.write("z_sup <- as.matrix(read.table(\"mc_supp_rate_graph.txt\", header=False, sep=\" \", skip=14))\n")


    f.write("\n")
    f.write("#Create the x and y vectors.\n")
    f.write("x <- c(" + )

    """#read the file
    z <- as.matrix(read.table("pathway_value_graph_1.txt", header=FALSE, sep=" ",skip=24))

    #cut off the last column, since it outputs with whitespace and confuses R
    z <- z[,-ncol(z)] #cutting off the last column, which is just whitespace

    #make x (and y) vectors
    x = seq(-20, 20, length.out = nrow(z))

    #make the levels
    levels_all <-seq(min(z),max(z),1.0)
    levels_high <- seq(min(z),max(z),0.25)
    levels2<-c(-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,6.25,6.5,6.75,7,7.25,7.5)

    #plot it
    filled.contour(x,x,z,xlim=range(x,finite=TRUE),ylim=range(x,finite=TRUE),levels_all,color.palette=topo.colors,plot.axes={contour(x,x,z,levels=levels_all,drawlabels=TRUE,axes=FALSE,frame.plot=FALSE,add=TRUE);axis(1);axis(2)})
    title(main="Ave. value of Monte Carlo sims in SWM v1.3 w/ feature transform      ", sub="(each point is the average value of 75 Monte Carlo Simulations)", xlab="Policy Parameter on Weather Severity", ylab="Policy Constant") 

    #save it
    pdf("MC_map_with_climbs_1_and_2.pdf")
    filled.contour(x,x,z,xlim=range(x,finite=TRUE),ylim=range(x,finite=TRUE),levels_all,color.palette=topo.colors,plot.axes={contour(x,x,z,levels=levels_all,drawlabels=TRUE,axes=FALSE,frame.plot=FALSE,add=TRUE);axis(1);axis(2)})
    title(main="Ave. value of Monte Carlo sims in SWM v1.3 w/ feature transform      ", sub="(each point is the average value of 75 Monte Carlo Simulations)", xlab="Policy Parameter on Weather Severity", ylab="Policy Constant") 
    dev.off()



    ########################
    # SUPPRESSION RATE MAP #
    ########################

    pdf("suppression_rate_map.pdf")

    z <- as.matrix(read.table("suppression_rate_map.txt", header=FALSE, sep=" ",skip=23))
    z <- z[,-ncol(z)] #cutting off the last column, which is just whitespace
    x = seq(-20, 20, length.out = nrow(z))

    levels_all <-seq(min(z),max(z),0.1)
    levels_sup <- seq(min(z),max(z),0.1)
    col1=2
    col2=4
    filled.contour(x,x,z,xlim=range(x,finite=TRUE),ylim=range(x,finite=TRUE),levels_all,color.palette=function(x)rev(heat.colors(x)),plot.axes={
    contour(x,x,z,levels=levels_sup,drawlabels=TRUE,axes=FALSE,frame.plot=FALSE,add=TRUE);
    axis(1);
    axis(2);})
    title(main="Suppr. Rate of Monte Carlo sims in SWM v1.3 w/ feature transform      ", sub="(each point is the average value of 75 Monte Carlo Simulations)", xlab="Policy Parameter on Weather Severity", ylab="Policy Constant") 

    dev.off()
    """