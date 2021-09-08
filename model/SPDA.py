import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import Parameter
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
import utils.utils as utils


class SPDAModel(nn.Module):
    def __init__(self, base_net='ResNet50', hidden_dim=1024, class_num=31):
        super(SPDAModel, self).__init__()
        # set base network
        self.base_network = backbone.network_dict[base_net]()
        self.hidden_layer_list = [nn.Linear(self.base_network.output_num(), hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.hidden_layer = nn.Sequential(*self.hidden_layer_list)
        self.classifier_layer_list = [nn.Linear(hidden_dim, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)

        # initialization
        self.hidden_layer[0].weight.data.normal_(0, 0.005)
        self.hidden_layer[0].bias.data.fill_(0.1)
        self.classifier_layer[0].weight.data.normal_(0, 0.01)
        self.classifier_layer[0].bias.data.fill_(0.0)

        # collect parameters
        self.parameter_list = [{"params": self.base_network.parameters(), "lr": 0.1},
                               {"params": self.hidden_layer.parameters(), "lr": 1},
                               {"params": self.classifier_layer.parameters(), "lr": 1}]

    def forward(self, source_inputs, target_inputs, source_label):
        source_features = self.base_network(source_inputs)
        target_features = self.base_network(target_inputs)

        source_hidden_features = self.hidden_layer(source_features)
        target_hidden_features = self.hidden_layer(target_features)

        source_outputs = self.classifier_layer(source_hidden_features)
        target_outputs = self.classifier_layer(target_hidden_features)

        dist = utils.moc_similarity(source_hidden_features, target_hidden_features)
        return source_outputs, target_outputs, dist

    def predict(self, target_inputs):
        target_features = self.base_network(target_inputs)
        target_hidden_features = self.hidden_layer(target_features)
        target_outputs = self.classifier_layer(target_hidden_features)
        return target_outputs

    def get_parameter_list(self):
            return self.parameter_list

    