from models import linknet


def loader(model, num_classes):
    if model == 'LinkNet18':
        return linknet.LinkNet18(num_classes)