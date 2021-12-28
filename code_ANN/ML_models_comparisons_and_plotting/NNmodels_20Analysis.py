import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from sklearn.metrics import mean_squared_error


from data_loader.data_loaders import ProcessParamDataLoaderTEST_20_Analysis
from utils import read_json
import numpy as np


def predict(model_name):

    cfg_fname = "saved_models/" + model_name + "/config.json"
    resume = "saved_models/" + model_name + "/model.pth"
    config_json = read_json(cfg_fname)
    config = ConfigParser(config_json, resume=resume)

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

    batch_size = 61
    data_loader = ProcessParamDataLoaderTEST_20_Analysis(config['data_loader']['args']['data_dir'],
                                                         norm_range_min,
                                                         norm_range_max,
                                                         data_loader_training.dataset,
                                                         batch_size,
                                                         shuffle=False,
                                                         validation_split=0.0,
                                                         training=False,
                                                         num_workers=0)

    # build model architecture
    model = config.init_obj('arch', module_arch)

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model.eval()
    print('model name', config['arch']['type'])

    # normalization range
    norm_range_min = config['data_loader']['args']['norm_range_min']
    norm_range_max = config['data_loader']['args']['norm_range_max']
    norm_range_diff = norm_range_max - norm_range_min

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):

            output = model(data)
            output_nonNormalized = ((output - norm_range_min)/norm_range_diff) * \
                (training_max_y-training_min_y) + training_min_y
            target_nonNormalized = ((target - norm_range_min)/norm_range_diff) * \
                (training_max_y-training_min_y) + training_min_y

            print("**********************************************************")
            print("selected model: ", model_name)
            print('predicted', output_nonNormalized)
            print('target', target_nonNormalized)

            absolute_error = abs(output_nonNormalized -
                                 target_nonNormalized)
            print('prediction error %', absolute_error)

            absolute_error = absolute_error.numpy()

            print("average prediction error", np.mean(absolute_error, axis=0))
            print("MSE: ", mean_squared_error(output, target))
            print("**********************************************************")

    return output_nonNormalized.numpy(), abs(output_nonNormalized - target_nonNormalized)
