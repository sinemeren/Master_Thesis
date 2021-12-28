import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser

from utils import read_json


def predict(model_name, data):

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

    training_max_x = data_loader_training.dataset.x_max
    training_min_x = data_loader_training.dataset.x_min

    '''
    print("training y min", training_min_y)
    print("training y max", training_max_y)
    print("training x min", training_min_x)
    print("training x max", training_max_x)
    '''
    norm_range_min = config['data_loader']['args']['norm_range_min']
    norm_range_max = config['data_loader']['args']['norm_range_max']

    # build model architecture
    model = config.init_obj('arch', module_arch)

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model.eval()

    # normalization range
    norm_range_min = config['data_loader']['args']['norm_range_min']
    norm_range_max = config['data_loader']['args']['norm_range_max']
    norm_range_diff = norm_range_max - norm_range_min

    data = ((data - training_min_x) / (training_max_x - training_min_x))

    data = torch.tensor(data)
    data = data.float()

    with torch.no_grad():

        output = model(data)

        output_nonNormalized = ((output - norm_range_min)/norm_range_diff) * \
            (training_max_y-training_min_y) + training_min_y

    return {'output': output_nonNormalized.numpy(), 'inputRange_min': training_min_x, 'inputRange_max': training_max_x}
