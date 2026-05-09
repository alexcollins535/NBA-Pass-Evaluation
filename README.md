**Spatiotemporal GNN for NBA Possession Value Modeling**

Utilized SportVU tracking data from the 2015-16 NBA season to answer the question, "Can we objectively grade an NBA ball-halndler's pass making decisions?"

**Approach** 

* Estimated Points Scored = Estimated Possession Value (EPV)
* Train a Gradient Boosting Classifier (GBC) to predict the likelihood of a pass being successful
* Train a Graph Neural Network (GNN) to estimate EPV given the current game state 
* Calculate expected EPV for a pass using: Expected EPV = Prob(Pass Success) * Future State EPV

* Generate counterfactual game states to evaluate all four receiver candidates

**Spatiotemporal GNN Architecture**

* In preprocessing, we restructured the player/ball tracking data as graphs, where each player and the ball were viewed as nodes with edges connecting each node to all other nodes.
* Additionally, we used global features (game clock, bonus status, etc.) to capture additional game context.
* LSTM (Long Short Term Memory) - for each sample frame, we feed the model the current game state, along with the previous 24 tracking frames. The LSTM layer compresses the temporal window of tracking data into a single vector.
* Static Features MLP - encodes the player shooting statistics.
* Fusion Layer - combines the static feature vector with the temporal window vector.
* GATv2Conv - runs graph convolution with attention twice to capture both how each player/ball node is affecting others and the overall offense/defense structure.
* Global Pooling - encodes the graph features into a single vector, along with the global features.
* MLP Head - Takes the pooled vector and learns the outcome value for the given possession.

**Counterfactual State Generation**

* For each detected pass, simulate a pass to all possible receivers on the offense.
* The future state is 0.5 seconds later.
* Project all players based on their current velocity.
* Project the ball to the receiver’s location.
* Generate the new 25 tracking frame window.
* For each simulated pass, feed the simulated state into the EPV and Pass Success models to get EPV and pass success likelihood.

**Results**

Baseline EPV Model: 
* Trained a baseline model on full possession data.
* Tested the intuition that less time left in the possession = fewer possible outcomes.
* Baseline Validation MSE: 1.2452
* MSE limiting to last 8 seconds of possession: 1.183
* MSE limiting to last 5 seconds of possession: 1.145
* MSE limiting to last 2 seconds of possession: 1.166
* This suggests that structured plays lead to more predictable outcomes, as unstructured plays will only be caught in the smaller late possession windows.

Fine-Tuned EPV Model:
* Fine tuned the model with an additional 3 epochs at a slower learning rate.
* Lowered MSE across Train, Validation, and Test data sets.
* Fine-tuned Validation MSE: 1.204
* MSE limiting to last 8 seconds of possession: 1.113
* MSE limiting to last 5 seconds of possession: 1.092
* MSE limiting to last 2 seconds of possession: 1.104

Pass Simulation:
* Evaluated three randomly selected games.
* Evaluated the average expected EPV of a pass selected vs. the average for the first, second, third, and fourth best options.
* Found that across all three games, the average expected EPV for a pass selected was near the average for the fourth best option.
* Separating out only the EPV score, not factoring in the likelihood of pass success, the average future EPV of a pass selected was near the average for the second best option.
* This suggests that players are making their passing decisions to maximize possession value regardless of the likelihood of pass success. 



