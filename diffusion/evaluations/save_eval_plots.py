import os

import numpy as np
import matplotlib.pyplot as plt

from diffusion.dist_util import read_text_prompts


def save_eval_plot(
    log_file,
    metric_name,
    frequency=None,
    out_file=None,
    max_viz_fid=1.0,
):
    if frequency is None:
        frequency = 1
    if out_file is None:
        out_file = os.path.join(os.path.dirname(log_file), metric_name + '.png')
    log = read_text_prompts(log_file)
    data_metric = None
    sample_metrics = []
    for text_line in log:
        if data_metric is None and (metric_name + ' Data') in text_line:
            data_metric = float(text_line.split(" ")[-1][:-1])
        elif metric_name in text_line and not (metric_name + ' Data') in text_line:
            metric_val = float(text_line.split(" ")[-1][:-1])
            sample_metrics.append(metric_val)
    sampling_times = frequency * (np.arange(len(sample_metrics)) + 1)
    plt.plot(sampling_times, sample_metrics)
    plt.xlabel('Training Steps')
    plt.ylabel(metric_name)
    if metric_name == 'FID':
        # avoid viz of very large fid points which make plot hard to read
        plt.ylim(plt.ylim()[0], min(max_viz_fid, plt.ylim()[1]))
    plt.savefig(out_file)
    plt.close()


def save_all_eval_plots(
    log_file,
    frequency=None,
    metrics=(
        'FID',
        'Sim',
        'Diversity',
        'R Prec Top 1',
        'R Prec Top 2',
        'R Prec Top 3',
    ),
    encoder_types=(
        'CLPP',
        'CLMP',
    ),
):
    for encoder_type in encoder_types:
        for metric in metrics:
            try:
                save_eval_plot(log_file, encoder_type + ' ' + metric, frequency=frequency)
            except:
                continue


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', default='', help='Name of training log file.')
    parser.add_argument('--frequency', default=1, type=int, help='Frequency of recording metric values.')
    args = parser.parse_args()

    save_all_eval_plots(args.log_file, frequency=args.frequency)
