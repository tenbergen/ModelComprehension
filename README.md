# ModelComprehension

## Authors
Proffesor Bastian Tenbergen <br>
Kritika Parajuli <br>
Thomas Marten <br>

## Description

This program's main goal is to take a pdf of software engineering diagrams and be able to pick out nodes/elements on these diagrams, find the center points of these nodes/elements, and put these center points through a Voronoi tessellation. Find the center points of the tessellation clusters and find the average distance standard deviation between these center points. 

* Give the program a pdf name, pdf file location and blur integer
* The program checks if the pdf is a valid pdf
* The program then converts the pdf to a png
* The program then converts the png to grayscale applies and a gaussian blur
* The program then applies a open cv inverse binary threshold
* The program then applies a open cv method for finding the contours of the png
* The program then finds the center point of every contour from the previous step found
* The program then uses the center points from the previous step in a Kmeans cluster algorithm
* The program then finds the center points of the K Means clusters
* The program then uses the center points from the previous step and creates a Voronoi tessellation graph from a custom Voronoi method
* The program then finds the center points of the Voronoi tessellation clusters
* The program then uses the center points from the previous step and calculates the average distance and standard deviation from the center points tessellation graph.

## Sources

