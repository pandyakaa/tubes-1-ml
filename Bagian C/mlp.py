import numpy as np 


class MyMlp(object):

    def __init__(self, input_layer, hidden_layer, output_layer):
        self.input = input_layer
        self.hidden = hidden_layer
        self.output = output_layer
        self.weights = []
        self.n_layer = len(input_layer) + len(hidden_layer) + len(output_layer)

    def initialize_weights(self) :
        pass 
    
    pass