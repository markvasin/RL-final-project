# Reinforcement Learning Project - Mountain Car
This repository contains the code and [report](https://github.com/markvasin/RL_mountain_car/blob/master/report/report.pdf) for the Reinforcement Learning project about [Mountain Car](https://en.wikipedia.org/wiki/Mountain_car_problem).

## Mountain Car
The goal of this problem is to get the under-power car to the top of the mountain. The states consisted of 2D continuous values of position (-1.2 to 0.6) and velocity (-0.07 to 0.07), and the available actions are move left (0), stay (1) and move right (2). For each time step, there is a reward (penalty) of -1 until the goal is reached.

![mountain-car](mountain-car.gif)

## Tools and Libraries
- [OpenAI Gym](https://gym.openai.com/)
- NumPy
- Scikit-learn

## Algorithms
- Q-Learning
- SARSA
- Radial Basis Functions

## Learning Curve
![plot](Learning%20plot.png)
