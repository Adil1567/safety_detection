
Hard Hat Workers - v13 augmented3x-HeadHelmetClasses-AccurateModel
==============================

This dataset was exported via roboflow.com on October 7, 2022 at 9:31 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 16867 images.
Workers are annotated in Pascal VOC format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -10 and +10 degrees
* Random brigthness adjustment of between -20 and +20 percent
* Random exposure adjustment of between -20 and +20 percent
* Random Gaussian blur of between 0 and 1 pixels


