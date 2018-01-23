#http://blog.kaggle.com/2017/12/22/carvana-image-masking-first-place-interview/
#https://github.com/asanakoy/kaggle_carvana_segmentation

from models import linknet, unet, unet_models, vgg_unet, unet1


def loader(model, num_classes):
    if model == 'LinkNet18':
        return linknet.LinkNet18(num_classes=num_classes) #albu
    if model == 'LinkNet34':
        return linknet.LinkNet34(num_classes=num_classes)
    if model == 'LinkNet50':
        return linknet.LinkNet50(num_classes=num_classes)
    
    if model == 'UNet11':
        return unet_models.UNet11(num_classes=num_classes, num_filters=32) #ternaus
    
    if model == 'UnetVgg11':
        return vgg_unet.UnetVgg11(n_classes=num_classes, num_filters=64, v=2)  #asanakoy
    if model == 'Vgg11a':
        return vgg_unet.Vgg11a(n_classes=num_classes, num_filters=32, v=1)    
    if model == 'Unet4':
        return unet.Unet4(n_classes=num_classes)
    if model == 'Unet5':
        return unet.Unet5(n_classes=num_classes)   
    if model == 'Unet7':
        return unet.Unet7(n_classes=num_classes)    
    if model == 'UNarrow':
        return unet.UNarrow(n_classes=num_classes)    
    if model == 'Unet':
        return unet.Unet(n_classes=num_classes) 
    
    if model == 'UNet256_3x3':
        return unet1.UNet256_3x3(in_shape=(3,256,256), num_classes=num_classes)