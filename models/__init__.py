import importlib

def create_model(config):
    lib = importlib.import_module('models.trainers.{}'.format(config.trainer))
    model = lib.Model(config)
    
    return model
