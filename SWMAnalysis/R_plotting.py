#R-plotting for SWMAnalyst
import subprocess

#creates "mc_surfaces_R_script.R"
def write_R_script_for_MC_surface_maps(x_vector, y_vector, title="", x_label="", y_label=""):
    
    f = open('mc_surfaces_R_script.R','w')
    f.write("#This script will plot the associated surface maps output by SWMAnalyst.\n")
    f.write("\n")

    f.write("#Read in the files\n")
    f.write("z_val <- as.matrix(read.table(\"mc_value_graph.txt\", header=FALSE, sep=\" \", skip=14))\n")
    f.write("z_var <- as.matrix(read.table(\"mc_variance_graph.txt\", header=FALSE, sep=\" \", skip=14))\n")
    f.write("z_sup <- as.matrix(read.table(\"mc_supp_rate_graph.txt\", header=FALSE, sep=\" \", skip=14))\n")


    f.write("\n")
    f.write("#Create the x and y vectors.\n")
    f.write("x <- c(" + str(x_vector).replace("[","").replace("]","") + ")\n")
    f.write("y <- c(" + str(y_vector).replace("[","").replace("]","") + ")\n")


    #pdf output of the value surface
    f.write("\n")
    f.write("pdf(\"Value Surface.pdf\")\n")
    f.write("filled.contour(x,y,z_val,")
    f.write("color.palette=topo.colors,")
    f.write("main=\"Value Surface: " + title + "\",")
    f.write("xlab=\"" + x_label + "\",")
    f.write("ylab=\"" + y_label + "\"")
    f.write(")\n")
    f.write("invisible(dev.off())\n")

    #pdf output of the variance surface
    f.write("\n")
    f.write("pdf(\"Variance Surface.pdf\")\n")
    f.write("filled.contour(x,y,z_var,")
    f.write("color.palette=topo.colors,")
    f.write("main=\"Variance Surface: " + title + "\",")
    f.write("xlab=\"" + x_label + "\",")
    f.write("ylab=\"" + y_label + "\"")
    f.write(")\n")
    f.write("invisible(dev.off())\n")

    #pdf output of the suppression rate surface
    f.write("\n")
    f.write("pdf(\"Suppression Rate Surface.pdf\")\n")
    f.write("filled.contour(x,y,z_sup,")
    f.write("color.palette=function(x)rev(heat.colors(x)),")
    f.write("main=\"Suppression Rate Surface: " + title + "\",")
    f.write("xlab=\"" + x_label + "\",")
    f.write("ylab=\"" + y_label + "\"")
    f.write(")\n")
    f.write("invisible(dev.off())\n")


    #png output of the value surface
    f.write("\n")
    f.write("png(\"Value Surface.png\")\n")
    f.write("filled.contour(x,y,z_val,")
    f.write("color.palette=topo.colors,")
    f.write("main=\"Value Surface: " + title + "\",")
    f.write("xlab=\"" + x_label + "\",")
    f.write("ylab=\"" + y_label + "\"")
    f.write(")\n")
    f.write("invisible(dev.off())\n")

    #png output of the variance surface
    f.write("\n")
    f.write("png(\"Variance Surface.png\")\n")
    f.write("filled.contour(x,y,z_var,")
    f.write("color.palette=topo.colors,")
    f.write("main=\"Variance Surface: " + title + "\",")
    f.write("xlab=\"" + x_label + "\",")
    f.write("ylab=\"" + y_label + "\"")
    f.write(")\n")
    f.write("invisible(dev.off())\n")

    #png output of the suppression rate surface
    f.write("\n")
    f.write("png(\"Suppression Rate Surface.png\")\n")
    f.write("filled.contour(x,y,z_sup,")
    f.write("color.palette=function(x)rev(heat.colors(x)),")
    f.write("main=\"Suppression Rate Surface: " + title + "\",")
    f.write("xlab=\"" + x_label + "\",")
    f.write("ylab=\"" + y_label + "\"")
    f.write(")\n")
    f.write("invisible(dev.off())\n")

#issues Rscript command to the OS to run the file "mc_surfaces_R_script.R"
def run_R_script_for_MC_surface_maps():
    subprocess.call(["Rscript","mc_surfaces_R_script.R"])

#does both functions
def create_R_plots(x_vector, y_vector, title="", x_label="", y_label=""):
    write_R_script_for_MC_surface_maps(x_vector, y_vector, title, x_label, y_label)
    run_R_script_for_MC_surface_maps()