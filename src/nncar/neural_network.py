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
    """Run the car's sensor readings through its network.

    Returns (accelerate, turn), both in [-1,1].

    This is the single hottest function in the project - it runs once per car
    per frame, and a training run evaluates it tens of millions of times - so
    it is written out flat rather than composed from the matrix helpers above.
    Layer.forward, activation, dot_product and add_matrices remain as the
    readable definition of what this computes, and tests/test_forward.py checks
    the two agree bit for bit on hundreds of random networks.

    Three things make the flat version faster without changing a single bit:
    the input is known to be a column vector, so the generic matrix code's
    innermost loop over columns is always a loop over one element; the
    multiply-accumulate, the bias and the activation happen in one pass instead
    of three, which avoids building two intermediate matrices per layer; and
    values move as plain floats rather than as one-element lists.

    The accumulation is a plain `for ... : total += w * v`. Using
    `sum(map(mul, ...))` measures faster still, but CPython 3.12 gave sum()
    compensated floating-point summation, so it does not produce identical
    results - and identical is the point.
    """
    values = [row[0] for row in car.inputs]

    for layer in car.network.layers:
        activated = []
        for weights,bias in zip(layer.weights,layer.bias):
            total = 0
            for weight,value in zip(weights,values):
                total += weight * value
            total += bias[0]

            # The same saturation clamp as Layer.activation: beyond +/-20 tanh
            # is 1 to within float precision, and the naive exponential form
            # would overflow not far past it.
            if total < -20:
                activated.append(-1)
            elif total > 20:
                activated.append(1)
            else:
                activated.append((math.exp(total) - math.exp(-total))
                                 / (math.exp(total) + math.exp(-total)))
        values = activated

    return values[0],values[1]

#: Default mutation step size, used when no sigma is supplied. The trainer
#: overrides it with a schedule; see nncar.ga.population.sigma_for.
mutation_rate = 0.05

#: Opponents in a race.
number_of_cars = 5