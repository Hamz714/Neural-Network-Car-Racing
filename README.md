# Neural Network Car Racing Game 🏎️🧠

A 2D car racing game built with **Python and Pygame**, where cars are controlled by a **custom-built neural network** instead of hard-coded logic.  
This project explores **neural networks, matrix operations, and AI decision-making** in a real-time game environment.

The AI learns to drive around the track using sensor-like inputs and neural network outputs to control steering and movement.

---

## 🚀 Features

- 🧠 Neural network–controlled cars (no ML libraries used)
- 📐 Manual implementation of:
  - Matrix multiplication
  - Weights and biases
  - Forward propagation
- 🎮 Fully playable Pygame racing game
- 🛣️ Multiple difficulty levels (Easy / Medium / Hard)

---

## 🧠 How the Neural Network Works

Each car is controlled by a feed-forward neural network implemented from scratch in `neural_network.py`.

### Inputs
The network receives numerical inputs representing:
- Distance from track borders
- Car speed
- Car orientation
- Progress along the track

### Outputs
The network outputs values that determine:
- Steering direction
- Acceleration and movement decisions

### Architecture
- Fully connected layers
- Randomly initialised weights and biases
- Activation function applied after each layer
- Forward propagation
- Natural Selection

No external machine learning libraries were used — this was intentional to gain a deep understanding of **how neural networks work internally**.

---

## 🗂️ Project Structure

```
Neural-Network-Car-Racing/
├── main.py                 # Game entry point
├── neural_network.py       # Neural network implementation
├── function.py             # Math and helper functions
├── variable.py             # Game constants, buttons, objects
├── window.py               # Rendering and UI logic
│
├── easy.txt                # Easy difficulty parameters
├── medium.txt              # Medium difficulty parameters
├── hard.txt                # Hard difficulty parameters
│
├── assets/
│   ├── track.png
│   ├── cars.png
│   ├── button images
│   └── rockit.mp3
│
├── LICENSE
└── .gitignore
```

---

## ▶️ How to Run

### Requirements
- Python 3.9 or later
- Pygame

### Install dependencies
```bash
pip install pygame
```

### Run the game
```bash
python main.py
```

---

## 🎯 Purpose of This Project

This project was built to:
- Learn neural networks **from first principles**
- Apply AI concepts in a real-time interactive system
- Strengthen understanding of:
  - Linear algebra
  - Matrix operations
  - Game loops and rendering
  - AI-driven decision making

No machine learning frameworks such as TensorFlow or PyTorch were used.

---

## 🧩 Possible Improvements

- Add training via genetic algorithms or reinforcement learning
- Improve physics and collision handling
- Refactor neural network for scalability

---

## 👤 Author

**Hamzah Ibrahim**

Built as a personal learning project exploring AI, neural networks, and game development from scratch.
