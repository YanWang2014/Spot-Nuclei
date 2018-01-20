from models import linknet, unet, unet_models, vgg_unet


def loader(model, num_classes):
    if model == 'LinkNet18':
        return linknet.LinkNet18(num_classes=num_classes) #albu
    
    if model == 'UNet11':
        return unet_models.UNet11(num_classes=num_classes) #ternaus
    
    if model == 'UnetVgg11':
        return vgg_unet.UnetVgg11(n_classes=num_classes)  #asanakoy
    if model == 'Vgg11a':
        return vgg_unet.Vgg11a(n_classes=num_classes)    
    if model == 'Unet4':
        return unet.Unet4(n_classes=num_classes)
    if model == 'Unet5':
        return unet.Unet5(n_classes=num_classes)   
    if model == 'Unet7':
        return unet.Unet7(n_classes=num_classes)    
    if model == 'UNarrow':
        return unet.UNarrow(n_classes=num_classes)    
    
