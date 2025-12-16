# Cart Pole V1 - REINFORCE Implementation

## Acknowledgments

This implementation is based on code originally created by Mehdi Shahbazi Khojasteh and distributed under the MIT License. The original source code has been used and modified in accordance with the terms of the MIT License.

### Original Copyright Notice
```
Copyright (c) 2024 Mehdi Shahbazi Khojasteh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Description

This repository contains a PyTorch implementation of the REINFORCE algorithm applied to the Cart Pole V1 environment from the Gymnasium library. The objective is to balance a pendulum mounted on a cart by applying horizontal forces to keep the pole upright.

## About REINFORCE

REINFORCE is a policy gradient reinforcement learning algorithm that optimizes policies to maximize cumulative discounted rewards. The algorithm works by:
- Increasing the probability of actions that lead to positive outcomes
- Decreasing the probability of actions that lead to negative outcomes
- Using neural networks for policy approximation
- Employing gradient ascent to optimize toward higher expected rewards

This approach enables the agent to learn effective policies through trial and error.

## Requirements

**Python Version:** 3.8.10 (tested on Windows 10)

**Dependencies:**
```
gymnasium==0.29.1
numpy==1.22.0
torch==2.0.1+cu118
```

*Note: This implementation uses the current Gymnasium library (not the deprecated Gym library) for optimal compatibility.*

## Usage

Pre-trained weights are provided in `./final_weights.pt`. You can immediately test the trained agent without retraining from scratch. The weights load automatically when you run the code, allowing you to visualize the agent's performance.

## Results

The implementation successfully trains an agent over 400 episodes, with training progress and reward curves demonstrating the learning process.

## License

This project is based on code licensed under the MIT License. See the original copyright notice above for complete license terms.
