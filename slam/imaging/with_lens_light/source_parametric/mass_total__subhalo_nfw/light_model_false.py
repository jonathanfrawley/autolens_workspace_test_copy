"""
__SLaM (Source, Light and Mass)__

This SLaM pipeline runner loads a strong lens dataset and analyses it using a SLaM lens modeling pipeline.

__THIS RUNNER__

Using two source pipelines, a light pipeline and a mass pipeline this runner fits `Imaging` of a strong lens system
where in the final phase of the pipeline:

 - The lens `Galaxy`'s `LightProfile`'s are modeled as an `EllipticalSersic` + `EllipticalExponential`, representing
   a bulge + disk model.
 - The lens `Galaxy`'s light matter mass distribution is fitted using the `EllipticalSersic` + EllipticalExponential of the
    `LightProfile`, where it is converted to a stellar mass distribution via constant mass-to-light ratios.
 - The lens `Galaxy`'s total mass distribution is modeled as an `EllipticalPowerLaw`.
 - The source `Galaxy`'s light is modeled parametrically using an `EllipticalSersic`.

This runner uses the SLaM pipelines:

 `slam/imaging/with_lens_light/source__parametric.py`.
 `slam/imaging/with_lens_light/light__parametric.py`.
 `slam/imaging/with_lens_light/mass__total.py`.

Check them out for a detailed description of the analysis!
"""
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
Specify the dataset type, label and name, which we use to determine the path we load the data from.
"""

dataset_name = "light_sersic__mass_sie__source_sersic"
pixel_scales = 0.1
dataset_path = path.join("dataset", "imaging", "with_lens_light", dataset_name)

"""
Using the dataset path, load the data (image, noise-map, PSF) as an `Imaging` object from .fits files.
"""
imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scales,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=pixel_scales, radius=3.0
)

imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

"""
__Settings__

The `SettingsPhaseImaging` describe how the model is fitted to the data in the log likelihood function.

These settings are used and described throughout the `autolens_workspace/examples/model` example scripts, with a 
complete description of all settings given in `autolens_workspace/examples/model/customize/settings.py`.

The settings chosen here are applied to all phases in the pipeline.
"""

settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid2D, sub_size=2)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

"""
__PIPELINE SETUP__

Transdimensional pipelines used the `SetupPipeline` object to customize the analysis performed by the pipeline,
for example if a shear was included in the mass model and the model used for the source galaxy.

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong 
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own setup object 
which is equivalent to the `SetupPipeline` object, customizing the analysis in that pipeline. Each pipeline therefore
has its own `SetupMass`, `SetupLightParametric` and `SetupSourceParametric` object.

The `Setup` used in earlier pipelines determine the model used in later pipelines. For example, if the `Source` 
pipeline is given a `Pixelization` and `Regularization`, than this `Inversion` will be used in the subsequent 
_SLaMPipelineLight_ and Mass pipelines. The assumptions regarding the lens light chosen by the `Light` object are 
carried forward to the `Mass`  pipeline.

The `Setup` again tags the path structure of every pipeline in a unique way, such than combinations of different
SLaM pipelines can be used to fit lenses with different models. If the earlier pipelines are identical (e.g. they use
the same `SLaMPipelineSource`. they will reuse those results before branching off to fit different models in the 
_SLaMPipelineLight_ and / or `SLaMPipelineMass` pipelines. 
"""

"""
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit and is used identically to the
hyper pipeline examples.

The `SetupHyper` object has a new input available, `hyper_fixed_after_source`, which fixes the hyper-parameters to
the values computed by the hyper-phase at the end of the Source pipeline. By fixing the hyper-parameter values in the
_SLaMPipelineLight_ and `SLaMPipelineMass` pipelines, model comparison can be performed in a consistent fashion.
"""

hyper = al.SetupHyper(
    hyper_search_no_inversion=af.DynestyStatic(maxcall=1),
    hyper_search_with_inversion=af.DynestyStatic(maxcall=1),
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=None,
    hyper_background_noise=None,
    hyper_fixed_after_source=True,
)

"""
__SLaMPipelineSourceParametric__

The parametric source pipeline aims to initialize a robust model for the source galaxy using `LightProfile` objects. 

 Source: This parametric source model is used by the SLaM Mass pipeline that follows, and thus sets the complexity of 
 the parametric source model of the overall fit. 
 
 For this runner the `SetupSourceParametric` customizes: 
 
 - That the bulge of the source `Galaxy` is fitted using an `EllipticalSersic`.
 - There is has no over `LightProfile` components (e.g. a disk, envelope)_.
"""

setup_source = al.SetupSourceParametric(
    bulge_prior_model=al.lp.EllipticalSersic,
    disk_prior_model=None,
    envelope_prior_model=None,
)

"""
 Light: The `LightProfile` used to model the lens `Galaxy`'s light. This is changed in the SLaM Light pipeline that follows.

 We use the default Light settings which model the lens galaxy's light as an `EllipticalSersic` bulge 
 and `EllipticalExponential` disk.
"""

setup_light = al.SetupLightParametric()

"""
 Mass: The `MassProfile` used to model the lens `Galaxy`'s mass. This is changed in the SLaM Mass pipeline that follows.
 
 Our experience with lens modeling has shown the `EllipticalIsothermal` profile is the simpliest model to fit that 
 provide a good fit to the majority of strong lenses.
 
 For this runner the `SetupMassProfile` customizes:

 - That the mass of the lens `Galaxy` is fitted using an `EllipticalIsothermal`.
 - That there is not `ExternalShear` in the mass model (this lens was not simulated with shear and we do not include 
 it in the mass model).
 - That the mass profile centre is (0.0, 0.0) (this assumption will be relaxed in the SLaM Mass Pipeline.
"""

setup_mass = al.SetupMassTotal(mass_prior_model=al.mp.EllipticalIsothermal)

"""We combine the `SetupSource`, `SetupLight`  and `SetupMass` above to create the SLaM parametric Source Pipeline."""

pipeline_source_parametric = al.SLaMPipelineSourceParametric(
    setup_mass=setup_mass, setup_light=setup_light, setup_source=setup_source
)

"""
__SLaMPipelineLight__

The `SLaMPipelineLightParametric` pipeline fits the model for the lens `Galaxy`'s bulge + disk light model. 

A full description of all options can be found ? and ?.

 The model used to represent the lens `Galaxy`'s light is input into `SLaMPipelineLightParametric` below and this runner uses an 
 `EllipticalSersic` + `EllipticalExponential` bulge-disk model in this example.
 
For this runner the `SLaMPipelineLightParametric` customizes:

 - The alignment of the centre and elliptical components of the bulge and disk.
 - If the disk is modeled as an `EllipticalExponential` or `EllipticalSersic`.

The `SLaMPipelineLightParametric` uses the mass model fitted in the previous `SLaMPipelineSource`'s.

The `SLaMPipelineLightParametric` and imported light pipelines determine the lens light model used in `Mass` pipelines.
"""

setup_light = al.SetupLightParametric(
    bulge_prior_model=al.lp.EllipticalSersic,
    disk_prior_model=al.lp.EllipticalExponential,
    align_bulge_disk_centre=False,
    align_bulge_disk_elliptical_comps=False,
)

pipeline_light = al.SLaMPipelineLightParametric(setup_light=setup_light)

"""
__SLaMPipelineMass__

The `SLaMPipelineMass` pipeline fits the model for the lens `Galaxy`'s total mass distribution. 

A full description of all options can be found ? and ?.

The model used to represent the lens `Galaxy`'s mass is input into `SLaMPipelineMassTotal` and this runner uses the 
default of an `EllipticalPowerLaw` in this example.

For this runner the `SLaMPipelineMass` customizes:

 - The `MassProfile` fitted by the pipeline.
 - If there is an `ExternalShear` in the mass model or not (this lens was not simulated with shear and 
   we do not include it in the mass model).
"""

setup_mass = al.SetupMassTotal(mass_prior_model=al.mp.EllipticalPowerLaw)

pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass, light_is_model=False)

"""
__SetupSubhalo__

The final pipeline fits the lens and source model including a `SphericalNFW` subhalo, using a grid-search of non-linear
searches. 

A full description of all options can be found ? and ?.

The models used to represent the lens `Galaxy`'s mass and the source are those used in the previous pipelines.

For this runner the `SetupSubhalo` customizes:

 - If the source galaxy (parametric or _Inversion) is treated as a model (all free parameters) or instance (all fixed) 
   during the subhalo detection grid search.
 - The NxN size of the grid-search.
"""

setup_subhalo = al.SetupSubhalo(source_is_model=True, number_of_steps=2)

"""
__SLaM__

We combine all of the above `SLaM` pipelines into a `SLaM` object.

The `SLaM` object contains a number of methods used in the make_pipeline functions which are used to compose the model 
based on the input values. It also handles pipeline tagging and path structure.
"""

slam = al.SLaM(
    path_prefix=path.join("slam", dataset_name),
    setup_hyper=hyper,
    pipeline_source_parametric=pipeline_source_parametric,
    pipeline_light_parametric=pipeline_light,
    pipeline_mass=pipeline_mass,
    setup_subhalo=setup_subhalo,
)

"""
__PIPELINE CREATION__

We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

We then run each pipeline, passing the results of previous pipelines to subsequent pipelines.
"""

from slam.imaging.with_lens_light.pipelines import source__parametric
from slam.imaging.with_lens_light.pipelines import light__parametric
from slam.imaging.with_lens_light.pipelines import mass__total
from slam.imaging.with_lens_light.pipelines import subhalo

source__parametric = source__parametric.make_pipeline(slam=slam, settings=settings)
source_results = source__parametric.run(dataset=imaging, mask=mask)


light__parametric = light__parametric.make_pipeline(
    slam=slam, settings=settings, source_results=source_results
)
light_results = light__parametric.run(dataset=imaging, mask=mask)


mass__total = mass__total.make_pipeline(
    slam=slam,
    settings=settings,
    source_results=source_results,
    light_results=light_results,
)
mass_results = mass__total.run(dataset=imaging, mask=mask)


subhalo = subhalo.make_pipeline_single_plane(
    slam=slam, settings=settings, mass_results=mass_results
)
subhalo.run(dataset=imaging, mask=mask)
