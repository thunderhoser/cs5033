"""Plots the ACF for a synthetic dataset, along with significance levels.

ACF = autocorrelation function
"""

import os
import errno
import argparse
import numpy
from scipy.stats import t as t_distribution
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot

MAX_LAG_TO_PLOT = 100
WHITE_NOISE_STDEV = 1.

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 600

DEFAULT_LINE_WIDTH = 3
MAIN_LINE_COLOUR = numpy.array([141, 160, 203], dtype=float) / 255
SIGNIFICANCE_LINE_COLOUR = numpy.array([252, 141, 98], dtype=float) / 255

ZERO_LINE_WIDTH = 1
ZERO_LINE_COLOUR = numpy.full(3, 0.)

FONT_SIZE = 25
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

NUM_POINTS_ARG_NAME = 'num_points'
LAG1_AUTOCORRELATION_ARG_NAME = 'lag1_autocorrelation'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

NUM_POINTS_HELP_STRING = 'Number of points in synthetic data series.'

LAG1_AUTOCORRELATION_HELP_STRING = (
    'Lag-1 autocorrelation of synthetic data series (red noise).  Must be in '
    'range 0...1.')

CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level.  This will be used to determine the significance '
    'threshold for autocorrelation.  Must be in range 0...1, so for example, if'
    ' you want 95%, make this 0.95.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.')

DEFAULT_OUTPUT_DIR_NAME = (
    '/home/ryan.lagerquist/Downloads/classes/cs5033_fall2018/'
    'autocorrelation_test')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_POINTS_ARG_NAME, type=int, required=False, default=1000,
    help=NUM_POINTS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAG1_AUTOCORRELATION_ARG_NAME, type=float, required=False,
    default=0.5, help=LAG1_AUTOCORRELATION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False,
    default=0.95, help=CONFIDENCE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_OUTPUT_DIR_NAME, help=OUTPUT_DIR_HELP_STRING)


def _generate_red_noise(num_points, lag1_autocorrelation):
    """Generate red-noise data series.

    This method is based on Equation 11.2.1 in the following lecture notes:

    https://atmos.washington.edu/~breth/classes/AM582/lect/lect8-notes.pdf

    N = number of points in series

    :param num_points: N in the above discussion.
    :param lag1_autocorrelation: Lag-1 autocorrelation.
    :return: red_noise_values: length-N numpy array of values in series.
    """

    white_noise_values = numpy.random.normal(
        loc=0., scale=WHITE_NOISE_STDEV, size=num_points)

    red_noise_values = numpy.full(num_points, numpy.nan)
    for i in range(num_points):
        if i == 0:
            red_noise_values[i] = white_noise_values[i]
            continue

        red_noise_values[i] = (
            lag1_autocorrelation * red_noise_values[i - 1] +
            numpy.sqrt(1 - lag1_autocorrelation ** 2) * white_noise_values[i]
        )

    return red_noise_values


def _find_significance_threshold(num_points, confidence_level):
    """Finds significance threshold for autocorrelation.

    :param num_points: Number of points in series.
    :param confidence_level: Confidence level (in range 0...1).  For example, if
        you want 95% confidence, make this 0.95.
    :return: min_absolute_autocorrelation: Minimum absolute autocorrelation.
        Any absolute autocorrelation > `min_absolute_autocorrelation` is
        statistically significant, according to the t-test performed by this
        method.
    """

    min_absolute_t_value = t_distribution.ppf(
        q=(1. - confidence_level) / 2, df=num_points - 2, loc=0., scale=1.)
    return numpy.power(
        float(num_points - 2) / min_absolute_t_value ** 2 + 1, -0.5)


def _compute_acf(values_in_series):
    """Computes autocorrelation function of values in series.

    L = number of lags

    :param values_in_series: 1-D numpy array with values in series.
    :return: autocorrelation_by_lag: length-L numpy array with autocorrelation
        for each lag.
    :return: lags: length-L numpy array of lags (integers).
    """

    autocorrelation_by_lag = numpy.correlate(
        values_in_series, values_in_series, mode='same')

    # Remove negative lags.
    lag_0_index = numpy.argmax(autocorrelation_by_lag)
    autocorrelation_by_lag = autocorrelation_by_lag[lag_0_index:]
    lags = numpy.linspace(
        0, len(autocorrelation_by_lag) - 1, num=len(autocorrelation_by_lag),
        dtype=int)

    # Divide by num points used to compute each autocorrelation.
    num_points_by_lag = len(values_in_series) - lags
    autocorrelation_by_lag = autocorrelation_by_lag / num_points_by_lag

    # Normalize so that lag-0 autocorrelation is 1 (true by definition).
    autocorrelation_by_lag = autocorrelation_by_lag / autocorrelation_by_lag[0]

    return autocorrelation_by_lag, lags


def _plot_series(values_in_series, title_string, output_file_name):
    """Plots data series.

    :param values_in_series: 1-D numpy array of values in series.
    :param title_string: Figure title.
    :param output_file_name: Path to output file (figure will be saved here).
    """

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))

    indices = numpy.linspace(
        0, len(values_in_series) - 1, num=len(values_in_series))
    axes_object.plot(
        indices, values_in_series, linestyle='solid', color=MAIN_LINE_COLOUR,
        linewidth=DEFAULT_LINE_WIDTH)

    pyplot.xlabel('Coordinate ($e.g.$, time)')
    pyplot.ylabel('Value')
    pyplot.xlim([indices[0], indices[-1]])

    pyplot.title(title_string)
    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


def _plot_acf(autocorrelation_by_lag, lags, min_absolute_autocorrelation,
              max_lag_to_plot, title_string, output_file_name):
    """Plots autocorrelation function.

    L = number of lags

    :param autocorrelation_by_lag: length-L numpy array with autocorrelation
        for each lag.
    :param lags: length-L numpy array of lags (integers).
    :param min_absolute_autocorrelation: Significance threshold.  This will be
        shown as a horizontal line in the plot.
    :param max_lag_to_plot: Maximum lag to plot.
    :param title_string: Figure title.
    :param output_file_name: Path to output file (figure will be saved here).
    """

    indices_to_plot = numpy.where(lags <= max_lag_to_plot)
    autocorrelation_by_lag = autocorrelation_by_lag[indices_to_plot]
    lags = lags[indices_to_plot]

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))

    axes_object.plot(
        lags, autocorrelation_by_lag, linestyle='solid', color=MAIN_LINE_COLOUR,
        linewidth=DEFAULT_LINE_WIDTH)

    these_x_values = numpy.array([lags[0], lags[-1]])
    these_y_values = numpy.array(
        [min_absolute_autocorrelation, min_absolute_autocorrelation])
    axes_object.plot(
        these_x_values, these_y_values, linestyle='dashed',
        color=SIGNIFICANCE_LINE_COLOUR, linewidth=DEFAULT_LINE_WIDTH)

    these_y_values = these_y_values * -1
    axes_object.plot(
        these_x_values, these_y_values, linestyle='dashed',
        color=SIGNIFICANCE_LINE_COLOUR, linewidth=DEFAULT_LINE_WIDTH)

    these_y_values = numpy.full(2, 0.)
    axes_object.plot(
        these_x_values, these_y_values, linestyle=':', color=ZERO_LINE_COLOUR,
        linewidth=ZERO_LINE_WIDTH)

    pyplot.xlabel('Lag')
    pyplot.ylabel('Autocorrelation')
    pyplot.xlim([0, max_lag_to_plot])

    pyplot.title(title_string)
    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


def _run(num_points, lag1_autocorrelation, confidence_level, output_dir_name):
    """Plots the ACF for a synthetic dataset, along with significance levels.

    This is effectively the main method.

    :param num_points: See documentation at top of file.
    :param lag1_autocorrelation: Same.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    assert num_points > 0
    assert lag1_autocorrelation > 0.
    assert lag1_autocorrelation < 1.
    assert confidence_level > 0.
    assert confidence_level < 1.

    try:
        os.makedirs(output_dir_name)
    except OSError as this_error:
        if this_error.errno == errno.EEXIST and os.path.isdir(output_dir_name):
            pass
        else:
            raise

    values_in_series = _generate_red_noise(
        num_points=num_points, lag1_autocorrelation=lag1_autocorrelation)
    min_absolute_autocorrelation = _find_significance_threshold(
        num_points=num_points, confidence_level=confidence_level)
    autocorrelation_by_lag, lags = _compute_acf(values_in_series)

    series_file_name = '{0:s}/data_series.jpg'.format(output_dir_name)
    series_title_string = (
        r'Red-noise time series with {0:d} points and $r_1$ = {1:.3f}'
    ).format(num_points, lag1_autocorrelation)
    _plot_series(
        values_in_series=values_in_series, title_string=series_title_string,
        output_file_name=series_file_name)

    acf_file_name = '{0:s}/autocorrelation_function.jpg'.format(output_dir_name)
    acf_title_string = (
        r'Red-noise ACF with {0:d} points and $r_1$ = {1:.3f} ... confidence '
        r'level = {2:.3f}'
    ).format(num_points, lag1_autocorrelation, confidence_level)
    _plot_acf(
        autocorrelation_by_lag=autocorrelation_by_lag, lags=lags,
        min_absolute_autocorrelation=min_absolute_autocorrelation,
        max_lag_to_plot=MAX_LAG_TO_PLOT, title_string=acf_title_string,
        output_file_name=acf_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        num_points=getattr(INPUT_ARG_OBJECT, NUM_POINTS_ARG_NAME),
        lag1_autocorrelation=getattr(
            INPUT_ARG_OBJECT, LAG1_AUTOCORRELATION_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
