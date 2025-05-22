# write a script to convert pytorch model to onnx model
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import onnx
from onnx import helper



if __name__ == '__main__':
 