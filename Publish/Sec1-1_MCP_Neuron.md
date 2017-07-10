
# Exploring Artificial Intelligence

A foray into artificial intelligence, with the help of math, history and Python  
by _Jonty Sinai_

## Section 1: Foundations of Machine Learning
## Part 1: The McCulloch-Pitts Neuron

We will begun by studying the earliest form of an artificial neuron: the **McCulloch-Pitts (MCP) Neuron**. The MCP Neuron is a simplified mathematical model of a biological neuron which ccan be used to construct Boolean logic gates.

Although the MCP neuron is rudimentary by today's standards, it formed an early and important stepping stone in the history of artificial neural networks. Frank Rosenblatt's _Perceptron_ and later _artificial neural networks_ both build on the fundamental ideas of the MCP neuron.

### A Very Brief Look at Neurons

We understand _neurons_ as electrically excitable, interconnected nerve cells in the brain which process and transmit information through electrical and chemical signals. The connections between neurons are known as _synapses_. When connected together, neurons and synapses from a _neural network_. Neurons consist of three main parts:

[_picture here_]

* The _cell body_ or _soma_: main part of the neuron which processes signals.
* _Dendrites_: branch-like shapes which receive signals from other neurons.
* _Axon_: a single nerve which sends signals to other neurons.

Thus a single neuron may receive many signals from other neurons via its dendrites. These signals are then combined and may fire off another signal from the neuron via its axon to other neurons.

[source: [wikipedia](https://en.wikipedia.org/wiki/Neuron)]


#### The MCP Model of an Artificial Neuron

The McCulloch-Pitts (MCP) model is the earliest mathematical representation of an artificial neuron. It was first proposed in 1943 by the neurophysiologist Walter S. McCulloch and the logician Walter Pitts. The MCP model abstracts the biological notion of a neuron as a mathematical model containing:

[_diagram here_]

* $m$ binary input signals, $\ x_1, x_2, ..., x_m \in \{0,1\}$.
* A set of binary _weights_ for each input, $\ w_1, w_2, ..., w_m \in \{-1,0,1\}$.
* Inputs with a weight of $1$ are called _excitatory_, while inputs with a weight of $-1$ are called _inhibitory_. 
* Inputs with a weight of $0$ do not contribute at all to the neuron. 
* An _activation function_, $\ f:\{0,1\}^{m} \to \{0,1\}$.
* A threshold value; an integer $\ t \in \mathbb{Z}$$^{*}$.
* An output signal, $\ y \in \{0,1\}$, such that $\ y = f(x_1, x_2, ..., x_m)$

$_{* \ \text{the keen mathematician will note that the threshold value is bounded by the number of input signals,} \ m \ \text{, so that} \ t \ \in \ [-m, m] \ \subset \ \mathbb{Z}}$

The logic is as follows:

* If the sum of the weighted inputs exceeds the threshold value, then the neuron is said to be _activated_ and the output signal is $1$. 
* Otherwise the neuron is _not activated_ and the output signal is $0$. 

Nowadays there are a variety of activation functions which are used to form a binary classifyer. In its original formulation, the activation function took the form of a _Heaviside step function_. The Heaviside step function matches the logic above by outputing $1$ when the neuron is activated; $0$ otherwise.

Ie.
  
$$
y = \left\{
\begin{array}{l}
1, \ \text{if} \ \sum_{i=1}^{m}w_{i}x_{i} \geq t,\\[3pt]
0, \ \text{otherwise}
\end{array}
\right.
$$

[[source](http://aishack.in/tutorials/artificial-neurons-mccullochpitts-model/)]

The following code chunk contains a Python representation of the MCP neuron. Notice how each logic gate is completely determined by its weights, which are predetermined by the user beforehand, and its threshold value. 

Using this model, McCulloch and Pitts showed (using some impressive logical calculus) that is was possible to construct the three basic _Boolean logic gates_: OR, AND and NOT$^{1}$. 

For an overview of logic gates, see [here](http://www.ee.surrey.ac.uk/Projects/CAL/digital-logic/gatesfunc/).


```python
import numpy as np
import pandas as pd

class MCPNeuron(object):
    """McCulloch-Pitts Neuron model
    
    Creates a logic gate using a set of weights and 
    an activation threshold. 
    
    Parameters
    ----------
        w : array-like, shape = [1, m_signals]
            Input weights, either -1, 0 or 1.
        t : int 
            Activation threshold.
    
    """
    
    def __init__(self, w = [1,1], t = 1):
        self.w = np.array(w)
        self.t = t
        
    
    def decide(self, message):
        """ Heaviside activation function.
        
        Returns 1 if the weighted sum of the input signals,
        passed as a message, exceeds the threshold value. 
        
        Returns 0, otherwise.
        
        Parameters
        ----------
            message : array-like, shape = [1, m_signals] 
                Array of input signals, either 0 or 1.
        
        Returns
        -------
            y : int
                Output signal, either 0 or 1.
        
        """
        
        x = message # consistency with function notation above
        sum_ = np.inner(self.w,x)
        
        if sum_ >= self.t:
            return 1
        else:
            return 0
        
        
    def table(self, in_signals, in_labels, out_label):
        """
        Generates a true-false table (dataframe) of n messages
        for a logic gate object constructed using the MCPNeuron 
        class, where a message is a 1-D array of m signals.
        
        Parameters
        ----------
            in_signals : array-like, shape = [n_messages, m_signals]
                Set of input signals, each 0 or 1.
            in_labels : list, length = m_signals
                Column labels, as strings, for the input signals
            out_label : str
                Column label for the output signal
            
        Returns
        -------
            table: dataframe, shape = [n_messages, m_signals + 1]
                Truth table showing relationship between in and out
                signals.
        
        """
        
        table = pd.DataFrame(in_signals, columns = in_labels)
        
        out_signals = []
        for row in in_signals:
            signal = self.decide(message = row)
            out_signals.append(signal)
            
        table[out_label] = pd.Series(out_signals)
        return table
        
```

**OR Gate**:

The OR gate is a logic gate which returns true (1) if at least one of its input signals is true (1).

* Weights: $\ w_1 = 1, w_2 = 1$
* Threshold: $\ t = 1$

| $x_1$      | $x_2$      | $y$       |
|:----------:|:----------:|:---------:|
| 0          | 0          | 0         |
| 0          | 1          | 1         |
| 1          | 0          | 1         |
| 1          | 1          | 1         |


```python
in_signals = np.array([[0,0], [0,1], [1,0], [1,1]])
in_labels = ['x1', 'x2']
out_label = 'y'

# instantiate OR gate as an MCP Neuron class
OR = MCPNeuron(w = [1,1], t = 1)

OR_table = OR.table(in_signals, in_labels = in_labels, out_label = out_label)

print(OR_table)

```

       x1  x2  y
    0   0   0  0
    1   0   1  1
    2   1   0  1
    3   1   1  1


**AND Gate**:

The AND gate is a logic gate which returns true (1) only if both of its input signals are true (1).

* Weights: $\ w_1 = 1, w_2 = 1$
* Threshold: $\ t = 2$

| $x_1$      | $x_2$      | $y$       |
|:----------:|:----------:|:---------:|
| 0          | 0          | 0         |
| 0          | 1          | 0         |
| 1          | 0          | 0         |
| 1          | 1          | 1         |


```python
in_signals = np.array([[0,0], [0,1], [1,0], [1,1]])

# instantiate AND gate as an MCP Neuron class
AND = MCPNeuron(w = [1,1], t = 2)

AND_table = AND.table(in_signals, in_labels = in_labels, out_label = out_label)

print(AND_table)


```

       x1  x2  y
    0   0   0  0
    1   0   1  0
    2   1   0  0
    3   1   1  1


**NOT Gate**:

The NOT gate inverts the signal of its input, so that if the input is true (1), then the output will be false (0) and vice-versa. In short, it _negates_ the input signal.

* Weights: $\ w_1 = -1$
* Threshold: $\ t = 0$

| $x_1$      | $y$       |
|:----------:|:---------:|
| 0          | 1         |
| 1          | 0         |


```python
NOT_signals = np.array([[0], [1]])

# instantiate NOT gate as an MCP Neuron class
NOT = MCPNeuron(w = [-1], t = 0)

NOT_table = NOT.table(NOT_signals, in_labels = ['x1'], out_label = 'y')

print(NOT_table)
```

       x1  y
    0   0  1
    1   1  0


**NAND Gate**:

The NAND gate is a logical composition of the AND gate followed by the NOT gate. Ie it negates the logic of the AND gate, returning true (1) when no more than one of its input signals is true (1). Ie. true is only returned when all input signals are false (0). 

* Weights: $\ w_1 = -1, w_2 = -1$
* Threshold: $\ t = -1$

| $x_1$      | $x_2$      | $y$       |
|:----------:|:----------:|:---------:|
| 0          | 0          | 1         |
| 0          | 1          | 1         |
| 1          | 0          | 1         |
| 1          | 1          | 0         |


```python
in_signals = np.array([[0,0], [0,1], [1,0], [1,1]])

# instantiate AND gate as an MCP Neuron class
NAND = MCPNeuron(w = [-1,-1], t = -1)

NAND_table = NAND.table(in_signals, in_labels = in_labels, out_label = out_label)

print(NAND_table)
```

       x1  x2  y
    0   0   0  1
    1   0   1  1
    2   1   0  1
    3   1   1  0


**NOR Gate**:

The NOR gate is a logical composition of the OR gate followed by the NOT gate. It negates the logic of the OR gate, returning true (1) only when none of the inputs are true (0). 

* Weights: $\ w_1 = -1, w_2 = -1$
* Threshold: $\ t = 0$

| $x_1$      | $x_2$      | $y$       |
|:----------:|:----------:|:---------:|
| 0          | 0          | 1         |
| 0          | 1          | 0         |
| 1          | 0          | 0         |
| 1          | 1          | 0         |


```python
in_signals = np.array([[0,0], [0,1], [1,0], [1,1]])

# instantiate AND gate as an MCP Neuron class
NOR = MCPNeuron(w = [-1,-1], t = 0)

NOR_table = NOR.table(in_signals, in_labels = in_labels, out_label = out_label)

print(NOR_table)
```

       x1  x2  y
    0   0   0  1
    1   0   1  0
    2   1   0  0
    3   1   1  0


**Challenge:** Can you find a set of weights and a threshold value to create the XOR (Exclusive OR) and the XNOR (Exclusive NOR) gates? Note that the XOR gate returns true if at least one but _not all_ of the inputs is true, while XNOR gate returns true if _all or none_ of the inputs are true. 

**Answer in my next post**

#### Academic Papers

1. W. S. McCulloch and W. Pitts. A logical calculus of the ideas immanent in nervous activity. The bulletin of mathematical biophysics, 5(4):115–133, 1943
