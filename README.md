# CPPN-WGAN-GP
This project is a WGAN-GP that utilizes a compositional pattern producing network as the generator.

See http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/ for info on CPPN-GANs.

This generator utilizes a Wasserstein loss with gradient pentalties and produces color-images.








The following python libraries are used: Keras with the TensorFlow, numpy, SciPy, and imageio.

The dataset currently being used is the IMM Face Database. Citation:
@TECHREPORT\{IMM2004-03160,
    author       = "M. M. Nordstr{\o}m and M. Larsen and J. Sierakowski and M. B. Stegmann",
    title        = "The {IMM} Face Database - An Annotated Dataset of 240 Face Images",
    year         = "2004",
    month        = "may",
    keywords     = "annotated image dataset, face images, statistical models of shape",
    number       = "",
    series       = "",
    institution  = "Informatics and Mathematical Modelling, Technical University of Denmark, {DTU}",
    address      = "Richard Petersens Plads, Building 321, {DK-}2800 Kgs. Lyngby",
    type         = "",
    url          = "http://www2.imm.dtu.dk/pubdb/p.php?3160",
    abstract     = "This note describes a dataset consisting of 240 annotated monocular images of 40 different human faces. Points of correspondence are placed on each image so the dataset can be readily used for building statistical models of shape. Format specifications and terms of use are also given in this note."
}
 
The gradient penalty is adapted from tensorpack. Citation:
            @misc{wu2016tensorpack,
		   title={Tensorpack},
		   author={Wu, Yuxin and others},
		   howpublished={\url{https://github.com/tensorpack/}},
		   year={2016}
		 }
