# RAPIDS Hyperparameter Optimization with Ray Tune

Tune is a scalable hyperparameter optimization (HPO) framework, built
on top of the Ray framework for distributed applications. It includes
modern, scalable HPO algorithms, such as HyperBand and PBT, and it
supports a wide variety of machine learning models.

Ray can run on the public cloud of your choosing, or on on-premise hardware. 

RAPIDS integrates smoothly with Ray Tune, using GPU acceleration to
speed up both model training and data prep by up to 40x over CPU-based
alternatives. For HPO sweeps, this can enable you to try more
parameter options and find more accurate classifiers.

## RAPIDS + Ray Tune sample notebooks

This sample notebook shows how to use Ray Tune to optimize XGBoost and
cuML Random Forest classifiers over a large dataset of airline arrival
times. By design, it is very similar to the RAPIDS examples provided
for other cloud and bring-your-own-cloud HPO offerings. As Tune offers
a variety of HPO algorithms, the sample includes utilities to compare
between them. (Note that the "best" HPO algorithm may be *very*
problem-dependent, so results are not fully generalizable.)

You need both Jupyter and RAPIDS 0.13 or later installed to begin. See
https://rapids.ai/start.html for instructions. We recommend using 0.14
nightly packages. For Ray, you should also install a few additional
packages.

```
pip install tabulate nb_black
pip install -U ray
pip install ray[tune]
pip install bayesian-optimization scikit-optimize
```


## For more details

See the blog post about RAPIDS on Ray Tune (coming soon!).

* For background on the Ray project: https://ray.io/
* To learn more about Ray Tune specifically: https://docs.ray.io/en/latest/tune.html
* cuML documentation for machine learning: https://docs.rapids.ai/api/cuml/nightly/
