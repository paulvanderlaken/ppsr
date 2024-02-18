# ppsr 0.0.5

Fixed the package alias breaking due to `roxygen` dependency

# ppsr 0.0.4

Removed parallel package from imports to fix bug on MacOS #32


# ppsr 0.0.3

No major changes.


# ppsr 0.0.2.1

## Major changes:
* Update to ggplot2 theme, removed irrelevant panel grid lines

## Bug fixes:
* #19 Parallelization should not now work. Library call to package instead of exporting functions by name.


# ppsr 0.0.2

## Major changes:

* Improved documentation, extended README, and more detailed Usage sections for functions
* Several user-irrelevant functions now no longer exported
* Support for (generalized) linear models for regression and binary classification (not multinomial)
* `available_algorithms()` and `available_evaluation_metrics()` to inspect options
* `visualize_pps()` now includes a `include_target` parameter to remove the target variable from the barplot visualization

## Bug fixes:

* Customizable axis labeling in the visualizations


# ppsr 0.0.1

## Major changes: 

* Adds a NEWS file
* Default color of negative correlations to dark red

## New features: 

* Parallel computing now optional in all `score_*()` and `visualize_*()` functions

## Bug fixes:

* related to classification models (#11 and #16)
* related to inconsistent text color (#21)


