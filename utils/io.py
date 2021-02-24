import os
import numpy as np


def print_format(widths, formaters, values, form_attr):
    return ' '.join([(form_attr % (width, form)).format(val) for (
        form, width, val) in zip(formaters, widths, values)])


def print_format_name(widths, values, form_attr):
    return ' '.join([(form_attr % (width)).format(val) for (width, val) in zip(
        widths, values)])

def print_metrics(header, metrics, banner=25):
    if len(metrics) == 17:
        print_metrics_ext(header, metrics)
        return
    print('\n', '*' * banner, header, '*' * banner)
    # metric_names_long = ['Recall', 'Precision', 'False Alarm Rate',
    #                      'GT Tracks', 'Mostly Tracked', 'Partially Tracked',
    #                      'Mostly Lost', 'False Positives', 'False Negatives',
    #                      'ID Switches', 'Fragmentations',
    #                      'MOTA', 'MOTP', 'MOTA Log']

    metric_names_short = ['Rcll', 'Prcn', 'FAR',
                          'GT', 'MT', 'PT', 'ML',
                          'FP', 'FN', 'IDs', 'FM',
                          'MOTA', 'MOTP', 'MOTAL']

    # metric_widths_long = [6, 9, 16, 9, 14, 17, 11, 15, 15, 11, 14, 5, 5, 8]
    metric_widths_short = [5, 5, 5, 4, 4, 4, 4, 6, 6, 5, 5, 5, 5, 5]

    metric_format_long = ['.1f', '.1f', '.2f',
                          '.0f', '.0f', '.0f', '.0f',
                          '.0f', '.0f', '.0f', '.0f',
                          '.1f', '.1f', '.1f']

    splits = [(0, 3), (3, 7), (7, 11), (11, 14)]
    print(' | '.join([print_format_name(
                     metric_widths_short[start:end],
                     metric_names_short[start:end], '{0: <%d}')
        for (start, end) in splits]))

    print(' | '.join([print_format(
                     metric_widths_short[start:end],
                     metric_format_long[start:end],
                     metrics[start:end], '{:%d%s}')
        for (start, end) in splits]))


def print_metrics_ext(header, metrics, banner=30):
    print('\n{} {} {}'.format('*' * banner, header, '*' * banner))
    # metric_names_long = ['IDF1', 'IDP', 'IDR',
    #                      'Recall', 'Precision', 'False Alarm Rate',
    #                      'GT Tracks', 'Mostly Tracked', 'Partially Tracked',
    #                      'Mostly Lost',
    #                      'False Positives', 'False Negatives', 'ID Switches',
    #                      'Fragmentations',
    #                      'MOTA', 'MOTP', 'MOTA Log']

    metric_names_short = ['IDF1', 'IDP', 'IDR',
                          'Rcll', 'Prcn', 'FAR',
                          'GT', 'MT', 'PT', 'ML',
                          'FP', 'FN', 'IDs', 'FM',
                          'MOTA', 'MOTP', 'MOTAL']

    # metric_widths_long = [5, 4, 4, 6, 9, 16,
    #   9, 14, 17, 11, 15, 15, 11, 14, 5, 5, 8]
    metric_widths_short = [5, 4, 4, 5, 5, 5, 4, 4, 4, 4, 6, 6, 5, 5, 5, 5, 5]

    metric_format_long = ['.1f', '.1f', '.1f',
                          '.1f', '.1f', '.2f',
                          '.0f', '.0f', '.0f', '.0f',
                          '.0f', '.0f', '.0f', '.0f',
                          '.1f', '.1f', '.1f']

    splits = [(0, 3), (3, 6), (6, 10), (10, 14), (14, 17)]

    print(' | '.join([print_format_name(
                     metric_widths_short[start:end],
                     metric_names_short[start:end], '{0: <%d}')
        for (start, end) in splits]))

    print(' | '.join([print_format(
                     metric_widths_short[start:end],
                     metric_format_long[start:end],
                     metrics[start:end], '{:%d%s}')
        for (start, end) in splits]))
    print('\n\n')