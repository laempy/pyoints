---
title: 'Pyoints: A Python package for point cloud, voxel and raster processing.'
tags:
  - Python
  - geoinformatics
  - remote sensing
  - point cloud analysis
  - raster analysis
authors:
  - name: Sebastian Lamprecht
    orcid: 0000-0002-8963-2762
    affiliation: 1
affiliations:
 - name: Trier University
   index: 1
date: 25 September 2018
bibliography: paper.bib
---


# Summary

The evolution of automated systems like autonomous robots and unmanned aerial
vehicles leads to manifold opportunities in science, agriculture and industry.
Remote sensing devices, like laser scanners and multi-spectral cameras, can be
combined with sensor networks to all-embracingly monitor a research object.

The analysis of such big data is based on geoinformatics and
remote sensing techniques. Today, next to physically driven approaches, machine learning
techniques are often used to extract relevant thematical information from the data sets.
Analysis requires a fusion of the data sets, which is made difficult conceptually
and technically by different data dimensions, data structures, and various
spatial, spectral, and temporal resolutions.

Today, various software to deal with these different data sources is available.
Software like GDAL [@GDAL] and OpenCV [@opencv_library] are intended for image
processing. Libraries, like PCL [@PCL], Open3D [@Open3D] and PDAL [@PDAL] focus
on 3D point cloud processing. Each of these software packages provide an API
specially designed to solve the problems of their field efficiently. When
developing algorithms for automated processing of various types of input data,
the differing APIs and programming languages of these software packages become
a drawback. To support fast algorithm development and a short familiarization,
a unified API would be desirable.

*Pyoints* is a python package to conveniently process and analyze point
cloud data, voxels, and raster images. It is intended to be used to support
the development of advanced algorithms for geo-data processing.

The fundamental idea of *Pyoints* is to overcome the conceptual distinction
between point clouds, voxel spaces, and rasters to simplify data analysis
and data fusion of variously structured data. Based on the assumption that any
geo-object can be represented by a point, a data structure has been designed
that provides a unified API for points, voxels, and rasters. Each data
structure maintains its characteristic features, to allow for intuitive use,
but all data is also considered as a two or three dimensional point cloud,
providing spatial indices that are required in many applications to speed up
spatial neighborhood queries.

During development, great emphasis was put on designing a powerful but simple
API while also providing solutions for most common problems. *Pyoints*
implements fundamental functions and some advanced algorithms for point cloud,
voxel, and raster data processing, like coordinate transformation, vector
algebra, point filters, and interpolation. *Pyoints* also provides a unified
API for loading and saving commonly used geo-data formats.

*Pyoints* was designed to support research activities and algorithm
development in the field of geoinformatics and remote sensing. Early versions of
the software have been used for some pre-studies at Trier University 
[@Lamprecht_2015a; @Lamprecht_2017a]. *Pyoints* is also used in the PANTHEON project [@PANTHEON] 
to monitor hazelnut orchards.

The source code of *Pyoints* is on GitHub [@Pyoints_GitHub]. The
documentation can be found on GitHub Pages [@Pyoints_docs].


# Acknowledgements

This work has been supported by the European Commission under the grant
agreement number 774571 Project PANTHEON.


# References
