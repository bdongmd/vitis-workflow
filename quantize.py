import os

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

def quant_model(float_model, quant_model, quant_dataset, batchsize, tfrec_dir, evaluate):
    '''Quantize the floating-point model
    Args:
        float_model (str): full path of floating-point model
        quant_model (str): full path of quantized model
        quant_dataset (str): full path quantized dataset
        batchsize (int): batchsize for quantization
        tfrec_dir (str): full path to folder containing TFRecord
        evaluate (bool): if or not evlaute quantized model
    '''
    head_tail = os.path.split(quant_model)
    os.midires(head_tail[0], exist_ok = True)
    
    # load float model
    float_model = load_model(float_model)

    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=quant_dataset)

    if evaluate:
        print('\n'+DIVIDER)
        print('Evaluating quantized model...')
        print(DIVIDER+'\n')

        quantized_model.compile(optimizer=Adam(),
                                loss='binary_crossentropy',
                                metrics=['accuray'])

        scores = quantized_model.evalaute(quant_dataset, verbose = 0)

        print('Quantized model accuracy: {0:.4f}'.format(scores[1]*100),'%')
        'Quantized model accuracy: {0:.4f}'.format(scores[1]*100),'%'

    return
