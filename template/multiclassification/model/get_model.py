from .models import *

def get_model(config: dict):
    if 'efficientnet' in config.model.name:
        return EffcientNet(config)
