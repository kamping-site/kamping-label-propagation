#!/usr/bin/env Rscript
library(plyr, warn.conflict = FALSE)
library(dplyr, warn.conflict = FALSE)

aggregator <- function(df) data.frame(AvgTime = mean(df$LabelProbTime, na.rm = TRUE))
load <- function(file) ddply(read.csv(file), "Graph", aggregator) 

mpi <- load("PlainMPI.csv")
kamping <- load("KaMPIngWrapper.csv")
kaminpar <- load("dKaMinParWrapper.csv")

cat("PlainMPI:\t\t", mean(mpi$AvgTime), "\n")
cat("KaMPIngWrapper:\t\t", mean(kamping$AvgTime), "\n")
cat("dKaMinParWrapper:\t", mean(kaminpar$AvgTime), "\n")

