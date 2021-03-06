"""
__Transdimensional Pipelines__

This transdimensional pipeline runner loads a strong lens dataset and analyses it using a transdimensional lens
modeling pipeline.

Using a pipeline composed of five phases this runner fits `Imaging` of a strong lens system, where in the final phase
of the pipeline:

 - The lens `Galaxy`'s light is modeled parametrically as an `EllipticalSersic`.
 - The lens `Galaxy`'s total mass distribution is modeled as an `EllipticalIsothermal`.
 - The source galaxy is modeled parametrically as an `EllipticalSersic`.

This uses the pipeline (Check it out full description of the pipeline):

 `autolens_workspace/transdimensional/imaging/with_lens_light/pipelines/light_parametric__mass_total__source_parametric.py`.
"""
from os import path
import autolens as al
import autolens.plot as aplt

dataset_name = "light_sersic__mass_sie__source_sersic"
pixel_scales = 0.2

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

"""Next, we create the mask we'll fit this data-set with."""

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

"""Make a quick subplot to make sure the data looks as we expect."""

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
__Pipeline_Setup__:

Pipelines use `Setup` objects to customize how different aspects of the model are fitted. 

First, we create a `SetupLightParametric` which customizes:

 - The `LightProfile`'s which fit different components of the lens light, such as its `bulge` and `disk`.
 - The alignment of these components, for example if the `bulge` and `disk` centres are aligned.
 - If the centre of the lens light profile is manually input and fixed for modeling.
 
In this example we fit the lens light as just one component, a `bulge` represented as an `EllipticalSersic`. We do 
not fix its centre to an input value. We have included options of `SetupLightParametric` with input values of
`None`, illustrating how it could be edited to fit different models.
"""

setup_light = al.SetupLightParametric(
    bulge_prior_model=al.lp.EllipticalSersic,
    disk_prior_model=None,
    envelope_prior_model=None,
    align_bulge_disk_centre=False,
    align_bulge_disk_elliptical_comps=False,
    light_centre=None,
)

"""
This pipeline also uses a `SetupMass`, which customizes:

 - The `MassProfile` used to fit the lens's total mass distribution.
 - If there is an `ExternalShear` in the mass model or not.
  - If the centre of the `MassProfile` is aligned with the centre of the `LightProfile`'s.
"""

setup_mass = al.SetupMassTotal(
    mass_prior_model=al.mp.EllipticalPowerLaw,
    with_shear=True,
    align_bulge_mass_centre=False,
)

"""
Next, we create a `SetupSourceParametric` which customizes the parametric source model in an identical way to how 
`SetupLightParametric` customizes the parametric lens light model.

This example uses a single `bulge` with an `EllipticalSersic` and we've included examples illustrating how this could 
be customized.
"""

setup_source = al.SetupSourceParametric(
    bulge_prior_model=al.lp.EllipticalSersic,
    disk_prior_model=None,
    envelope_prior_model=None,
    align_bulge_disk_centre=False,
    align_bulge_disk_elliptical_comps=False,
    light_centre=None,
)

"""
_Pipeline Tagging_

The `Setup` objects are input into a `SetupPipeline` object, which is passed into the pipeline and used to customize
the analysis depending on the setup. This includes tagging the output path of a pipeline. For example, if `with_shear` 
is True, the pipeline`s output paths are `tagged` with the string `with_shear`.

This means you can run the same pipeline on the same data twice (e.g. with and without shear) and the results will go
to different output folders and thus not clash with one another!

The `path_prefix` below specifies the path the pipeline results are written to, which is:

 `autolens_workspace/output/transdimensional/dataset_type/dataset_name` 
 `autolens_workspace/output/transdimensional/imaging/light_sersic__mass_sie__source_sersic/`

The redshift of the lens and source galaxies are also input (see `examples/model/customize/redshift.py`) for a 
description of what inputting redshifts into **PyAutoLens** does.
"""

setup = al.SetupPipeline(
    path_prefix=path.join("transdimensional", dataset_name),
    redshift_lens=0.5,
    redshift_source=1.0,
    setup_light=setup_light,
    setup_mass=setup_mass,
    setup_source=setup_source,
)

"""
__Pipeline Creation__

To create a pipeline we import it from the pipelines folder and run its `make_pipeline` function, inputting the 
`Setup` and `SettingsPhase` above.
"""

from pipelines import light_parametric__mass_total__source_parametric

pipeline = light_parametric__mass_total__source_parametric.make_pipeline(
    setup=setup, settings=settings
)

"""
__Pipeline Run__

Running a pipeline is the same as running a phase, we simply pass it our lens dataset and mask to its run function.
"""

pipeline.run(dataset=imaging, mask=mask)
