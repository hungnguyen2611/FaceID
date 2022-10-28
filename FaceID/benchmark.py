import net
import torch
import os
from face_alignment import align
import numpy as np
import time



adaface_models = {
    'ir_18':"pretrained/adaface_ir18_vgg2.ckpt"
}


def load_pretrained_model(architecture='ir_18'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor



if __name__ == '__main__':
    start = time.time()
    model = load_pretrained_model('ir_18')
    end = time.time()
    print("Loading weight time took {} sec".format(end-start))
    start = time.time()
    for i in range(10):
        feature, norm = model(torch.randn(1,3,112,112))
    end = time.time()
    print("Warm up time took {} sec".format((end-start)/10))
    start = time.time()
    for i in range(100):
        input = torch.randn(112,112,3)
        bgr_tensor_input = to_input(input)
        feature, _ = model(bgr_tensor_input)
    end = time.time()
    print("Predicting took avg {} sec".format((end-start)/100))



