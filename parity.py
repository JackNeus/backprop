import random
from backprop import init_new, train_net, forward_cycle, save_data, load_data

#N = 2, 3  10000
#N = 4

N = 5

UNITS = [N, N, 1] #First index is input, last is output 

def run(input_layer):
  return forward_cycle(list(input_layer))[-1][:-1] #Return only last layer (minus bias)

cases = []
def gen_all_cases(partial):
  if len(partial) == N:
    cases.append(partial)
    return
  a = list(partial) + [0]
  b = list(partial) + [1]
  gen_all_cases(a)
  gen_all_cases(b)   

def gen_case():
  case = []
  for i in range(N):
    case.append(random.randint(0, 1))
  return case

def scale(n):
  if n == 0:
    return -1
  return 1

def scale_list(n):
  return [scale(p) for p in n]

def ans(input_data):
  num_ones = 0
  for val in input_data:
    if val == 1:
      num_ones += 1
  return [int(num_ones % 2 == 1)]

init_new(UNITS, learn_rate = .1)

gen_all_cases([])

def train(n, verbosity = 0):
  print("Training...")

  num_training_rounds = n
  update_inc = max(100, n // 100)
  for i in range(0, num_training_rounds):
    if verbosity > 0:
      if i % update_inc == 0:
        p = int(i / n * 100)
        print(str(p) + '% complete. (' + str(i) + ')')
    random.shuffle(cases)
    for case in cases:
      train_net(list(case), scale_list(ans(case)))

  print(str(num_training_rounds) + " training rounds complete.")

def test(n, verbosity = 0):
  print("Testing...")
  trials = n
  correct = 0

  for i in range(0, trials):
    input_data = gen_case()
    r = run(input_data)[0]
    if verbosity > 0:
      print(input_data, ans(input_data)[0], r)
    if ans(input_data)[0] == 1 and r >= 0.9 or ans(input_data)[0] == 0 and r <= -0.9:
      correct += 1

  print (str(correct) + "/" + str(trials) + " correct.") 

load_data('parity' + str(N) + '.net')
#train(1000, 1)
test(1000)

