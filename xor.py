import random
from backprop import init_new, train_net, forward_cycle, save_data, load_data

UNITS = [2, 2, 1] #First index is input, last is output 
learning_rate = 0.1
momentum_rate = 0.9

def run(input_layer):
  return forward_cycle(input_layer)[-1][:-1] #Return only last layer (minus bias)

def scale(n):
  if n == 0:
    return -1
  return 1

def ans(input):
  return input[0] ^ input[1]

def train(n):
  print("Training...")

  num_training_cases = n
  for i in range(0, num_training_cases):
    a = random.randint(0, 1)
    b = random.randint(0, 1)
    train_net([scale(a), scale(b)], [scale(ans([a, b]))])

  print(str(num_training_cases) + " training sessions complete.")

def test(n, verbosity = 0):
  print("Testing...")
  trials = 0
  correct = 0

  for i in range(0, n):
    a = random.randint(0, 1)
    b = random.randint(0, 1)
    r = run([scale(a), scale(b)])[0]
    if verbosity > 0:
      print(a, b, ans([a, b]), r)
    trials += 1
    if ans([a, b]) == 1 and r >= 0.9 or ans([a, b]) == 0 and r <= -0.9:
      correct += 1

  print (str(correct) + "/" + str(trials) + " correct.") 
#save_data('xor.net')

init_new(UNITS, learning_rate, momentum_rate) #init blank
train(100)
test(100)
save_data('xor.net')
load_data('xor.net')
test(100)
train(10000)
test(10000)
save_data('xor.net')
