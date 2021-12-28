import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

from data_loader.data_loaders import ProcessParamDataLoaderTEST

import numpy as np
import pandas as pd


def main(config):
    logger = config.get_logger('test')

    print(config['data_loader']['type'])
    # setup data_loader instances

    data_loader_training = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        config['data_loader']['args']['file_name'],
        512,
        0,
        0,
        0,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=0
    )
    training_max_y = data_loader_training.dataset.y_max
    training_min_y = data_loader_training.dataset.y_min

    norm_range_min = config['data_loader']['args']['norm_range_min']
    norm_range_max = config['data_loader']['args']['norm_range_max']

    batch_size = 8
    data_loader = ProcessParamDataLoaderTEST('data/',
                                             norm_range_min,
                                             norm_range_max,
                                             data_loader_training.dataset,
                                             batch_size,
                                             shuffle=False,
                                             validation_split=0.0,
                                             training=False,
                                             num_workers=2)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print('model name', config['arch']['type'])
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    # normalization range
    norm_range_min = config['data_loader']['args']['norm_range_min']
    norm_range_max = config['data_loader']['args']['norm_range_max']
    norm_range_diff = norm_range_max - norm_range_min

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)

            output = model(data)

            output_nonNormalized = ((output - norm_range_min)/norm_range_diff) * \
                (training_max_y-training_min_y) + training_min_y
            target_nonNormalized = ((target - norm_range_min)/norm_range_diff) * \
                (training_max_y-training_min_y) + training_min_y

            print('predicted', output_nonNormalized)
            print('target', target_nonNormalized)

            absolute_error = abs(output_nonNormalized -
                                 target_nonNormalized)
            print('prediction error %', absolute_error)

            absolute_error = absolute_error.numpy()

            print("average prediction error", np.mean(absolute_error, axis=0))

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    df = pd.DataFrame(output_nonNormalized.numpy())  # convert to a dataframe
    df.to_excel("output.xlsx", index=False)  # save to file


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
