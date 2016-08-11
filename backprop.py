import random
from math import exp
import datetime
import json
from sys import exit

weight_file = None

UNITS = [1, 1]
learning_rate = .5
momentum_rate = .5
data_file = None

NET = [] #Weights - 3D array (layer, unit, weights to units in next layer)
prev_adjusts = [] #For momentum term

def init_new(config = [1, 1], learn_rate = .5, m_rate = .5, filename = None):
  global UNITS, learning_rate, momentum_rate, data_file
  UNITS = config
  learning_rate = learn_rate
  momentum_rate = m_rate
  data_file = filename

  if filename is not None:
    load_data(filename)
  else:
    for l in range(0, len(UNITS) - 1): #Layer
      layer_weights = [] 
      adj_row = []
      for u in range(0, UNITS[l] + 1): #Unit (Include extra for bias unit (last one))
        weights = []
        for w in range(0, UNITS[l + 1]):
          weights.append(random.random() * 4 * (random.randint(0, 1) * 2 - 1))
        layer_weights.append(weights)
        adj_row.append([0] * UNITS[l + 1])
      NET.append(layer_weights)
      prev_adjusts.append(adj_row)

def load_data(filename = None):
  if filename is None:
    print("No filename provided. Quitting.")
    exit()

  try: 
    f = open(filename, "r")
  except IOError:
    print("Error opening file. Quitting.")
    exit()

  global NET
  global prev_adjusts

  try:
    data = f.read().split('\n')
    NET = json.loads(data[0])
    prev_adjusts = json.loads(data[1])
  except Exception as e:
    print(e)
    print("Error loading net from file. Quitting.")
    exit()

  new_config = []
  for row in NET:
    new_config.append(len(row) - 1)
  new_config.append(len(NET[-1][0])) #Number of output nodes

  if new_config != UNITS:
    print("Structure of loaded net does not match original configuration. Quitting.")
    exit()

  f.close()
  print("Net loaded.")

def save_data(filename = None):
  if filename is None:
    filename = str(datetime.datetime.now()).replace(' ', 'T').replace('.','').replace(':','')[:-6] + '.NET'

  try:
    f = open(filename, "w")
    f.write(dump_obj(NET)+'\n')
    f.write(dump_obj(prev_adjusts)+'\n')
    f.close()
  except Exception as e:
    print(e)
    print("Error saving net to " + filename + ". Would you like to dump the net?")
    ans = raw_input()
    if ans.lower() in "yes"[:len(ans)]:
      print(dump_obj(NET))
    print("Quitting.")
    exit()

  print("Net saved.")

def dump_obj(data, none_value = "None"):
  if data is None:
    return none_value
  return json.dumps(data)

#BACK PROP LOGIC

def o(x):
  return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
def der_o(x):
  return 4 / (exp(x) + exp(-x)) ** 2
 
def forward_cycle(input_layer):
  input_layer += [1] #Add bias
  outputs = [input_layer]
  for l in range(1, len(UNITS)):
    layer = [0] * UNITS[l] + [1]
    for u in range(0, UNITS[l]):
      net = 0;
      for uu in range(0, UNITS[l - 1] + 1):
        net += outputs[l - 1][uu] * NET[l - 1][uu][u]
      layer[u] = o(net)
    outputs.append(layer)
  return outputs #Note that this includes biases at the end

def train_net(input_layer, expected):
  a = 1

  results = forward_cycle(input_layer)
  deltas = []
    
  #Output Layer
  layer = []
  for u in range(UNITS[-1]):
    delta = (expected[u] - results[-1][u]) * der_o(results[-1][u])
    layer.append(delta)
  deltas.append(layer)

  #Subsequent Layers, starting closest to output layer
  for l in range(len(UNITS) - 2, -1, -1):
    layer = []
    for u in range(UNITS[l]):
      error_term = 0
      for w in range(len(NET[l][u])):
        error_term += NET[l][u][w] * deltas[-1][w];
      delta = der_o(results[l][u]) * error_term
      layer.append(delta)
    deltas.append(layer)
  
  deltas = deltas[::-1]
  
  #print(deltas)
  
  #Now we calculate the change in weights 
  adjusts = []

  for l in range(0, len(UNITS) - 1):
    row = []
    for i in range(0, len(results[l])): #src (include bias)
      layer = []
      o = results[l][i]
      for j in range(0, UNITS[l + 1]): #dest
        d = deltas[l + 1][j]
        dw = learning_rate * d * o
        layer.append(dw)
      row.append(layer)
    adjusts.append(row)

  global prev_adjusts

  #Apply change in weights
  for l in range(0, len(adjusts)):
    for i in range(0, len(adjusts[l])):
      for j in range(0, len(adjusts[l][j])):
        NET[l][i][j] += adjusts[l][i][j] #+ momentum_rate * prev_adjusts[l][i][j]

  prev_adjusts = adjusts
    
