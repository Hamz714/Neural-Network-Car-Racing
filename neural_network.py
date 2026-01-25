import random,math
from copy import deepcopy

class Layer:
    def __init__(self,columns,rows):
        self.weights = random_matrix(rows,columns)
        self.bias = random_matrix(rows,1)

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

    def mutate(self):
        self.weights = multiply(self.weights)
        self.bias = multiply(self.bias)

    def tanh(self,input):
        return (math.exp(input) - math.exp(-input)) / (math.exp(input) + math.exp(-input))

class Network:
    def __init__(self):
        self.layers = [Layer(6,12),Layer(12,10),Layer(10,8),Layer(8,2)]

    def mutate(self):
        for layer in self.layers:
            layer.mutate()
        return self

def random_normal():
    input1 = random.random()
    input2 = random.random()
    output = math.sqrt(-2 * math.log(input1)) * math.cos(2 * math.pi * input2)
    return output

def random_matrix(rows,columns):
    matrix = [[random_normal() for i in range(columns)] for j in range(rows)]
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

def multiply(matrix):
    for row in range(len(matrix)):
        for column in range(len(matrix[0])):
            matrix[row][column] += mutation_rate * random_normal()
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