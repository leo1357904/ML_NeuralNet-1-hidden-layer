import sys
import numpy as np
import scipy.special as sp

class NeuralNet:
  """Code author: Ting-Sheng Lin (tingshel@andrew.cmu.edu)"""
  def __init__(self):
    self.lenOfLabels = 10
    self.weight = []
    self.metrix = {}
    self.trainEntropy = []
    self.testEntropy = []
  

  def __fileToDatapoints(self, train_file):
    data_rows = np.genfromtxt(train_file, delimiter=',')
    tmp_X, tmp_Y = data_rows[:,1:], data_rows[:,0]
    X, Y = [], []
    for i in range(len(tmp_X)):
      X.append(np.append(tmp_X[i], [1])) # add bias term at the end
      Y.append(int(tmp_Y[i]))
    return X, Y

  # init_flag=1 generate weights with random uniform distribution in [-0.1, 0.1], bias term 0.
  # init_flag=2 generate weights with all 0.
  #
  # every weight structure in weight:
  # [[w11, w21, w31],
  #  [w12, w22, w32],
  #  [b1,  b2,  b3 ]] #last row is bias term row
  def __initWeights(self, unitsOfLayers, init_flags):
    weights = []
    for i in range(len(unitsOfLayers) - 1):
      thetas = []
      for j in range(unitsOfLayers[i] + 1):
        if j == unitsOfLayers[i]:
          theta = [0] * unitsOfLayers[i + 1] # bias term
        else:
          theta = [0] * unitsOfLayers[i + 1] if init_flags == 2 else np.random.uniform(-0.1, 0.1, unitsOfLayers[i + 1])
        thetas.append(theta)
      weights.append(np.array(thetas))
    return weights

  def __sigmoid(self, a):
    return 1 / (1 + np.exp(-a))

  def __d_sigmoid(self, a):
    return self.__sigmoid(a) * (1 - self.__sigmoid(a))

  def __forwardComputing(self, x, y):
    a = x.dot(self.weights[0])
      
    sigmoid_v = np.vectorize(self.__sigmoid)
    z = sigmoid_v(a)
    z = np.append(z, [1])

    b = z.dot(self.weights[1])

    y_hat = sp.softmax(b)

    J = -np.log(y_hat[y])

    return [a, z, b, y_hat, J]


  def __backwardComputing(self, x, y, forwardObject):
    a, z, b, y_hat, J = forwardObject

    # J = yi*log(yi_hat)
    # dJ/dy_hat * dy_hat/db = y_hat - y
    g_b = np.subtract(y_hat, np.array([0] * y + [1] + [0] * (self.lenOfLabels - y - 1)))

    g_beta = np.outer(z, g_b)

    beta_star = self.weights[1][:-1]
    g_z = g_b.dot(beta_star.T)

    d_sigmoid_v = np.vectorize(self.__d_sigmoid) # d sigmoid/d a
    g_a = np.multiply(g_z, d_sigmoid_v(a)) # np.multiply [a1, a2], [b1, b2] -> [a1b1, a2b2]

    g_alpha = np.outer(x, g_a)

    return [g_alpha, g_beta]


  def train(self, train_file, test_file, num_epoch, hidden_units, init_flags, learning_rate):
    self.trainEntropy, self.testEntropy, self.metrix = [], [], {}
    X, Y = self.__fileToDatapoints(train_file) # X includes bias term
    testX, testY = self.__fileToDatapoints(test_file)

    # unitsOfLayers is a list of units_number in each layer (include x and labels)
    # **Change this line if having multi hidden-layers 
    unitsOfLayers = [len(X[0]) - 1] + [hidden_units] + [self.lenOfLabels] 
    self.weights = np.array(self.__initWeights(unitsOfLayers, init_flags))
    
    for _ in range(num_epoch): # every epoch
      for i in range(len(X)): # train weights
        a, z, b, y_hat, J = self.__forwardComputing(X[i], Y[i])
        g_alpha, g_beta = self.__backwardComputing(X[i], Y[i], [a, z, b, y_hat, J])

        self.weights[0] = np.subtract(self.weights[0], np.dot(learning_rate, g_alpha))
        self.weights[1] = np.subtract(self.weights[1], np.dot(learning_rate, g_beta))

      totalJ = 0
      for i in range(len(X)): # calculate mean entropy for train data
        a, z, b, y_hat, J = self.__forwardComputing(X[i], Y[i])
        totalJ += J
      self.trainEntropy.append(totalJ / len(X))

      totalJ = 0
      for i in range(len(testX)): # calculate mean entropy for test data
        a, z, b, y_hat, J = self.__forwardComputing(testX[i], testY[i])
        totalJ += J
      self.testEntropy.append(totalJ / len(testX))          


  def test(self, test_file, predict_file, error_name):
    X, Y = self.__fileToDatapoints(test_file)
    
    predictions_str = ''
    error_count = 0
    for i in range(len(X)):
      a, z, b, y_hat, J = self.__forwardComputing(X[i], Y[i])
      prediction = int(np.argmax(y_hat))
      predictions_str += f'{prediction}\n'
      if prediction != Y[i]:
        error_count += 1
    self.metrix[error_name] = error_count / len(X)
    f_out = open(predict_file,"w+")
    f_out.write(predictions_str)


  def metrics(self, metrics_file):
    out_str = ''
    for i in range(len(self.trainEntropy)):
      out_str += f"epoch={i + 1} crossentropy(train): {self.trainEntropy[i]}\n"
      out_str += f"epoch={i + 1} crossentropy(test): {self.testEntropy[i]}\n"

    for name, error_rate in self.metrix.items():
      out_str += f'error({name}): {error_rate:.6f}\n'

    f_out = open(metrics_file,"w+")
    f_out.write(out_str)
      



if __name__ == '__main__':
  train_input = sys.argv[1]
  test_input = sys.argv[2]
  train_out = sys.argv[3]
  test_out = sys.argv[4]
  metrics_out = sys.argv[5]
  num_epoch = int(sys.argv[6])
  hidden_units = int(sys.argv[7])
  init_flags = int(sys.argv[8])
  learning_rate = float(sys.argv[9])
  
  nn = NeuralNet()
  nn.train(train_input, test_input, num_epoch, hidden_units, init_flags, learning_rate)
  nn.test(train_input, train_out, 'train')
  nn.test(test_input, test_out, 'test')
  nn.metrics(metrics_out)
