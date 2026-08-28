import random,math
from copy import deepcopy

class Layer:
    """One fully connected layer: weights, biases, and a tanh activation."""

    def __init__(self,columns,rows,rng=None):
        self.weights = random_matrix(rows,columns,rng)
        self.bias = random_matrix(rows,1,rng)

    def forward(self,input):
        return add_matrices(dot_product(self.weights,input),self.bias)
    
    def activation(self,input):
        for row in range(len(input)):
            for column in range(len(input[0])):
                if input[row][column] < -20:
                    input[row][column] = -1
                elif input[row][column] > 20:
                    input[row][column] = 1
                else:
                    input[row][column] = self.tanh(input[row][column])
        return input

    def mutate(self,sigma=None,rng=None):
        self.weights = multiply(self.weights,sigma,rng)
        self.bias = multiply(self.bias,sigma,rng)

    def tanh(self,input):
        return (math.exp(input) - math.exp(-input)) / (math.exp(input) + math.exp(-input))

class Network:
    """A 6-12-10-8-2 feed-forward network: 320 weights and biases.

    Inputs are five sensor distances plus the car's velocity; outputs are
    acceleration and steering, both squashed to [-1,1] by tanh.
    """

    def __init__(self,rng=None):
        self.layers = [Layer(6,12,rng),Layer(12,10,rng),Layer(10,8,rng),Layer(8,2,rng)]

    def copy(self):
        """An independent deep copy, so mutating a child never touches a parent."""
        return deepcopy(self)

    def mutate(self,sigma=None,rng=None):
        """Perturb every parameter by sigma * N(0,1), in place."""
        for layer in self.layers:
            layer.mutate(sigma,rng)
        return self

def random_normal(rng=None):
    """A draw from N(0,1) by the Box-Muller transform.

    Passing an rng (any random.Random) keeps a caller's stream independent of
    the global one, which is what lets training runs be reproducible while the
    game carries on using the module-level random. Omitting it uses the global
    stream exactly as before.

    Box-Muller produces two independent normals per pair of uniforms; only the
    cosine term is kept. Reclaiming the sine would halve the calls but shift
    every seeded value in the project, so it is deliberately left as is.
    """
    source = rng if rng is not None else random
    input1 = source.random()
    while input1 == 0.0:  # log(0) would raise; probability ~2^-53 per draw
        input1 = source.random()
    input2 = source.random()
    output = math.sqrt(-2 * math.log(input1)) * math.cos(2 * math.pi * input2)
    return output

def random_matrix(rows,columns,rng=None):
    matrix = [[random_normal(rng) for i in range(columns)] for j in range(rows)]
    return matrix

def dot_product(matrix1,matrix2):
    new_matrix = [[0 for i in range(len(matrix2[0]))] for j in range(len(matrix1))]
    for row in range(len(new_matrix)):
        for column in range(len(new_matrix[0])):
            for iteration in range(len(matrix1[0])):
                new_matrix[row][column] += matrix1[row][iteration] * matrix2[iteration][column]
    return new_matrix

def add_matrices(matrix1,matrix2):
    new_matrix = [[0 for i in range(len(matrix1[0]))] for j in range(len(matrix1))]
    for row in range(len(matrix1)):
        for column in range(len(matrix1[0])):
            new_matrix[row][column] = matrix1[row][column] + matrix2[row][column]
    return new_matrix

def multiply(matrix,sigma=None,rng=None):
    """Add sigma * N(0,1) to every element, in place."""
    step = mutation_rate if sigma is None else sigma
    for row in range(len(matrix)):
        for column in range(len(matrix[0])):
            matrix[row][column] += step * random_normal(rng)
    return matrix

def forward_propagation(car):
    output = car.inputs
    for layer in car.network.layers:
        output = layer.forward(output)
        output = layer.activation(output)
    return output[0][0],output[1][0]

def best_networks(NPC_cars):
    NPC_cars.sort(reverse=True, key=lambda car : car.score)
    NPC_cars = NPC_cars[:number_of_cars//10]
    return NPC_cars

def mutation(NPC_cars,NPC):
    new_NPC_cars = []
    for car in NPC_cars:
        network = deepcopy(car.network)
        new_NPC_cars.append(NPC(deepcopy(car.start_x),deepcopy(car.start_y),deepcopy(network)))
        for i in range(9):
            network = deepcopy(deepcopy(car.network).mutate())
            new_NPC_cars.append(NPC(deepcopy(car.start_x),deepcopy(car.start_y),deepcopy(network)))
    return new_NPC_cars
                
mutation_rate = 0.05
number_of_cars = 5