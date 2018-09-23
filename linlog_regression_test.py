"""Simple experiment with linear or logistic regression.

This script has only one predictor variable (x) and one target variable (y).
"""

import os
import errno
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

X_MINIMUM = -10.
X_MAXIMUM = 10.
INITIAL_WEIGHT = 0.
NUM_TRAINING_POINTS = 1000
NUM_TESTING_POINTS = 1000
DEFAULT_NOISE_STDEV_FOR_LINEAR = 5.
DEFAULT_NOISE_STDEV_FOR_LOGISTIC = 0.2

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 600

LINE_WIDTH = 4
LINE_COLOUR = numpy.array([252, 141, 98], dtype=float) / 255

MARKER_TYPE = 'o'
MARKER_SIZE = 8
MARKER_EDGE_WIDTH = 1
MARKER_COLOUR = numpy.array([141, 160, 203], dtype=float) / 255

FONT_SIZE = 25
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

TRUE_SLOPE_ARG_NAME = 'true_slope'
TRUE_INTERCEPT_ARG_NAME = 'true_intercept'
NOISE_STDEV_ARG_NAME = 'noise_standard_deviation'
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
LEARNING_RATE_ARG_NAME = 'learning_rate'
LOGISTIC_ARG_NAME = 'logistic'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

TRUE_SLOPE_HELP_STRING = (
    'True slope (dy/dx).  This will be used to construct the training and '
    'testing sets.')

TRUE_INTERCEPT_HELP_STRING = (
    'True intercept (y at x = 0).  This will be used to construct the training '
    'and testing sets.')

NOISE_STDEV_HELP_STRING = (
    'Standard deviation of Gaussian noise.  This will be added to y when '
    'constructing the training and testing sets.')

NUM_ITERATIONS_HELP_STRING = (
    'Number of iterations.  This is the number of times that the weights will '
    'be updated.')

LEARNING_RATE_HELP_STRING = (
    'Learning rate.  This is alpha in the weight-update rule.')

LOGISTIC_HELP_STRING = (
    'Boolean flag (0 or 1).  If 1, this script will do logistic regression.  If'
    ' 0, will do linear regression.')

OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.')

DEFAULT_OUTPUT_DIR_NAME = (
    '/home/ryan.lagerquist/Downloads/classes/cs5033_fall2018/'
    'linlog_regression_test')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TRUE_SLOPE_ARG_NAME, type=float, required=False, default=10.,
    help=TRUE_SLOPE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + TRUE_INTERCEPT_ARG_NAME, type=float, required=False, default=-5.,
    help=TRUE_INTERCEPT_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NOISE_STDEV_ARG_NAME, type=float, required=False, default=-1,
    help=NOISE_STDEV_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ITERATIONS_ARG_NAME, type=int, required=False, default=10000,
    help=NUM_ITERATIONS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LEARNING_RATE_ARG_NAME, type=float, required=False, default=0.001,
    help=LEARNING_RATE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LOGISTIC_ARG_NAME, type=int, required=False, default=0,
    help=LOGISTIC_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_OUTPUT_DIR_NAME, help=OUTPUT_DIR_HELP_STRING)


def _inverse_logit(input_values):
    """Implements the inverse logit function, defined below.

    f(x) = exp(x) / [1 + exp(x)]

    :param input_values: numpy array of logits.
    :return: output_values: equivalent-size numpy of non-logits.
    """

    return numpy.exp(input_values) / (1 + numpy.exp(input_values))


def _binary_xentropy(actual_values, predicted_values):
    """Computes binary cross-entropy.

    :param actual_values: numpy array of actual values.
    :param predicted_values: equivalent-size numpy array of predicted values.
    :return: cross_entropy: Binary cross-entropy.
    """

    return -numpy.mean(
        actual_values * numpy.log2(predicted_values) +
        (1. - actual_values) * numpy.log2(1. - predicted_values)
    )


def _mean_squared_error(actual_values, predicted_values):
    """Computes mean squared error.

    :param actual_values: numpy array of actual values.
    :param predicted_values: equivalent-size numpy array of predicted values.
    :return: mean_squared_error: Mean squared error.
    """

    return numpy.mean((predicted_values - actual_values) ** 2)


def _generate_data(
        num_points, true_slope, true_intercept, noise_standard_deviation,
        do_logistic):
    """Generates dataset (either training or testing data).

    N = number of examples

    :param num_points: Number of examples (data points).
    :param true_slope: See documentation at top of file.
    :param true_intercept: Same.
    :param noise_standard_deviation: Same.
    :param do_logistic: Same.
    :return: x_values: length-N numpy array of x-values (predictor values).
    :return: y_values: length-N numpy array of y-values (target values).
    """

    x_values = numpy.random.uniform(
        low=X_MINIMUM, high=X_MAXIMUM, size=num_points)
    noise_values = numpy.random.normal(
        loc=0., scale=noise_standard_deviation, size=num_points)
    y_values = true_intercept + true_slope * x_values

    if do_logistic:
        y_values = _inverse_logit(y_values)
        y_values = numpy.round(y_values + noise_values)
        y_values[y_values < 0] = 0
        y_values[y_values > 1] = 1
    else:
        y_values = y_values + noise_values

    return x_values, y_values


def _train_model(
        x_values, y_values, num_iterations, learning_rate, do_logistic):
    """Trains linear- or logistic-regression model.

    N = number of examples

    :param x_values: length-N numpy array of x-values (predictor values).
    :param y_values: length-N numpy array of y-values (target values).
    :param num_iterations: See documentation at top of file.
    :param learning_rate: Same.
    :param do_logistic: Same.
    :return: weights: length-2 numpy array, where weights[0] is the estimated
        bias (beta_0) and weights[1] is the estimated slope (beta_1).
    """

    weights = numpy.full(2, INITIAL_WEIGHT)

    for i in range(num_iterations):
        predicted_y_values = weights[0] + weights[1] * x_values

        if do_logistic:
            predicted_y_values = _inverse_logit(predicted_y_values)
            error = _binary_xentropy(
                actual_values=y_values, predicted_values=predicted_y_values)
        else:
            error = _mean_squared_error(
                actual_values=y_values, predicted_values=predicted_y_values)

        first_gradient = 2 * numpy.mean(predicted_y_values - y_values)
        second_gradient = 2 * numpy.mean(
            x_values * (predicted_y_values - y_values))

        print (
            'Iteration {0:d} of {1:d} ... beta_0 = {2:.4e} ... beta_1 = {3:.4e}'
            ' ... error = {4:.4e}'
        ).format(i + 1, num_iterations, weights[0], weights[1], error)

        weights[0] -= learning_rate * first_gradient
        weights[1] -= learning_rate * second_gradient

    return weights


def _plot_results(x_values, y_values, weights, do_logistic, title_string,
                  output_file_name):
    """Plots results of linear or logistic regression.

    N = number of examples

    :param x_values: length-N numpy array of x-values (predictor values).
    :param y_values: length-N numpy array of y-values (actual target values).
    :param weights: length-2 numpy array, where weights[0] is the bias (beta_0)
        and weights[1] is the slope (beta_1).
    :param do_logistic: See documentation at top of file.
    :param title_string: Figure title.
    :param output_file_name: Path to output file (the figure will be saved
        here).
    """

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))

    axes_object.plot(
        x_values, y_values, linestyle='None', marker=MARKER_TYPE,
        markerfacecolor=MARKER_COLOUR, markeredgecolor=MARKER_COLOUR,
        markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH)

    if do_logistic:
        x_values_in_line = numpy.linspace(X_MINIMUM, X_MAXIMUM, num=1000)
    else:
        x_values_in_line = numpy.array([X_MINIMUM, X_MAXIMUM])

    y_values_in_line = weights[0] + weights[1] * x_values_in_line
    if do_logistic:
        y_values_in_line = _inverse_logit(y_values_in_line)

    axes_object.plot(
        x_values_in_line, y_values_in_line, color=LINE_COLOUR,
        linewidth=LINE_WIDTH)

    pyplot.xlabel(r'Predictor ($x$)')
    pyplot.ylabel(r'Target ($y$)')

    pyplot.title(title_string)
    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    pyplot.savefig(output_file_name, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


def _run(true_slope, true_intercept, noise_standard_deviation, num_iterations,
         learning_rate, do_logistic, output_dir_name):
    """Simple experiment with linear or logistic regression.

    This is effectively the main method.

    :param true_slope: See documentation at top of file.
    :param true_intercept: Same.
    :param noise_standard_deviation: Same.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param do_logistic: Same.
    :param output_dir_name: Same.
    """

    try:
        os.makedirs(output_dir_name)
    except OSError as this_error:
        if this_error.errno == errno.EEXIST and os.path.isdir(output_dir_name):
            pass
        else:
            raise

    # If noise standard deviation <= 0, set to default.
    if noise_standard_deviation <= 0:
        if do_logistic:
            noise_standard_deviation = DEFAULT_NOISE_STDEV_FOR_LOGISTIC + 0.
        else:
            noise_standard_deviation = DEFAULT_NOISE_STDEV_FOR_LINEAR + 0.

    training_x_values, training_y_values = _generate_data(
        num_points=NUM_TRAINING_POINTS, true_slope=true_slope,
        true_intercept=true_intercept,
        noise_standard_deviation=noise_standard_deviation,
        do_logistic=do_logistic)

    weights = _train_model(
        x_values=training_x_values, y_values=training_y_values,
        num_iterations=num_iterations, learning_rate=learning_rate,
        do_logistic=do_logistic)
    print SEPARATOR_STRING

    predicted_training_y_values = weights[0] + weights[1] * training_x_values
    base_title_string = r'Iterations = {0:d} ... $\alpha$ = {1:.4e}'.format(
        num_iterations, learning_rate)

    if do_logistic:
        predicted_training_y_values = _inverse_logit(
            predicted_training_y_values)
        training_error = _binary_xentropy(
            actual_values=training_y_values,
            predicted_values=predicted_training_y_values)
        training_title_string = '{0:s} ... training x-entropy = {1:.4e}'.format(
            base_title_string, training_error)
    else:
        training_error = _mean_squared_error(
            actual_values=training_y_values,
            predicted_values=predicted_training_y_values)
        training_title_string = '{0:s} ... training MSE = {1:.4e}'.format(
            base_title_string, training_error)

    training_figure_file_name = '{0:s}/training_results.jpg'.format(
        output_dir_name)
    _plot_results(
        x_values=training_x_values, y_values=training_y_values, weights=weights,
        do_logistic=do_logistic, title_string=training_title_string,
        output_file_name=training_figure_file_name)

    testing_x_values, testing_y_values = _generate_data(
        num_points=NUM_TESTING_POINTS, true_slope=true_slope,
        true_intercept=true_intercept,
        noise_standard_deviation=noise_standard_deviation,
        do_logistic=do_logistic)

    predicted_testing_y_values = weights[0] + weights[1] * testing_x_values
    if do_logistic:
        predicted_testing_y_values = _inverse_logit(predicted_testing_y_values)
        testing_error = _binary_xentropy(
            actual_values=testing_y_values,
            predicted_values=predicted_testing_y_values)
        testing_title_string = '{0:s} ... testing x-entropy = {1:.4e}'.format(
            base_title_string, testing_error)
    else:
        testing_error = _mean_squared_error(
            actual_values=testing_y_values,
            predicted_values=predicted_testing_y_values)
        testing_title_string = '{0:s} ... testing MSE = {1:.4e}'.format(
            base_title_string, testing_error)

    testing_figure_file_name = '{0:s}/testing_results.jpg'.format(
        output_dir_name)
    _plot_results(
        x_values=testing_x_values, y_values=testing_y_values, weights=weights,
        do_logistic=do_logistic, title_string=testing_title_string,
        output_file_name=testing_figure_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        true_slope=getattr(INPUT_ARG_OBJECT, TRUE_SLOPE_ARG_NAME),
        true_intercept=getattr(INPUT_ARG_OBJECT, TRUE_INTERCEPT_ARG_NAME),
        noise_standard_deviation=getattr(
            INPUT_ARG_OBJECT, NOISE_STDEV_ARG_NAME),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        learning_rate=getattr(INPUT_ARG_OBJECT, LEARNING_RATE_ARG_NAME),
        do_logistic=bool(getattr(INPUT_ARG_OBJECT, LOGISTIC_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
