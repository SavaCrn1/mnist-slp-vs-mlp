# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 19:21:54 2026

@author: savac
"""
## IMPORT PACKAGES
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#%% Import Data

from sklearn.datasets import fetch_openml
mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
    )

X = mnist.data                  
y = mnist.target.astype(int)    

#%% DATA PREPARATION
''' 
70,000 examples
784 features per image
10 target values (0-9)
256 pixel values (0-255)
'''

# Noramlize X inputs
X = X/256

X_train, X_test, y_train, y_test = train_test_split(
    X,y, 
    test_size=.2,
    random_state=42, 
    stratify = y)
#%%
'''
PART 1.
In-class Single layer Perceptron model modified to fit MINST dataset.
3 Learning Rates tested for training accuracy with the number of epochs and 
np.random.seed() held constant.
'''

learning_rates = [.001, 0.01, 1.0]        # Rate at which model learns
epochs = 10          # Titrations (number of times model gets to learn) 

#%% Initialize Dicts to store results for visualization
errors_by_rate = {}
test_accuracy_by_rate = {}

#%% Train and Test with different learning rates
for learning_rate in learning_rates:
    print(f"-----------------Learning rate: {learning_rate}-----------------")

    # Establish same conditions for each learning rate
    np.random.seed(42)
    W = np.random.uniform(0, 1, (10, 784))
    b = np.random.uniform(0, 1, 10)
    
    epoch_errors = [] # Initialize List to store errors into errors_by_rate for vis
    
    for epoch in range(epochs):
        
        total_errors = 0
        
        for x, correct_digit in zip(X_train, y_train):
            
            # Create Target Output
            # Example: digit 3 -> [0 0 0 1 0 0 0 0 0 0]        
            y_target = np.zeros(10)
            y_target[correct_digit] = 1
            
            # Compute weighted sum
            z = W @ x + b
            
            # Compute output using step function
            y_output = np.where(z >= 0, 1, 0)
    
            # Compute error 
            delta = y_target - y_output
            
            # Update weights
            W = W + learning_rate * np.outer(delta, x) 
            # np.outer calculates weighted change from the error (delta) times x
            
            # Update bias
            b = b + learning_rate * delta
            
            # Count classification errors
            if np.any(delta != 0):
                total_errors += 1
            
        # Store values for visual
        epoch_errors.append(total_errors)
        error_rate = round((100 -(total_errors/70000*100)),2)
        print(
            f"Epoch {epoch + 1}: "
            f"{total_errors} patterns with errors ("
            f"{error_rate} % accuracy)"
        )
        
        errors_by_rate[learning_rate] = epoch_errors
        
    # TESTING
    
    correct = 0 
    
    for x, correct_digit in zip(X_test, y_test):
        
        z = W @ x + b
        # Choose neuron with largest weighted sum
        predicted_digit = np.argmax(z)
        
        if predicted_digit == correct_digit:
            correct += 1
            
    accuracy = correct / len(X_test) * 100
    
    print()
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print()

#%% Errors across epochs by learning rate VISUAL

'''
See how quickly each learning rate decreases its number of errors across epochs.
'''

epoch_numbers = range(1, epochs + 1)

colors = {
    0.001: "blue",
    0.01: "orange",
    1.0: "red"
}

plt.figure(figsize=(9, 6))

for learning_rate in learning_rates:
    plt.plot(
        epoch_numbers,
        errors_by_rate[learning_rate],
        color=colors[learning_rate],
        marker="o",
        linewidth=2,
        label=f"Learning rate = {learning_rate}"
    )

plt.xlabel("Epoch")
plt.ylabel("Number of Errors")
plt.title("Training Errors Across Epochs")
plt.xticks(epoch_numbers)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
#%%
'''
PART 2.
MultiLayer Perceptron
'''
from sklearn.neural_network import MLPClassifier

hidden_layer_size_list = [32, 64, 128, (64,32)]

for size in hidden_layer_size_list:
    print(f"--------------------Hidden Layer Size: {size}--------------------")
    model = MLPClassifier(
        hidden_layer_sizes=size,
        max_iter=100,
        random_state=42,
    )
    
    model.fit(X_train, y_train)
    
    training_accuracy = model.score(X_train, y_train) # Used AI to get ".score" function for accuracies
    testing_accuracy = model.score(X_test, y_test)
    
    print(f"Training accuracy: {training_accuracy * 100:.2f}%")
    print(f"Testing accuracy:  {testing_accuracy * 100:.2f}%")
    print()

#%%
'''
PART 3.
Model Evaluation and Comparison.
'''
# Re running SLP with learning rate 0.001
np.random.seed(42)
W = np.random.uniform(0, 1, (10, 784))
b = np.random.uniform(0, 1, 10)

y_pred_slp = [] #List to derive a y_pred value for cm
for epoch in range(epochs):
    
    total_errors = 0
    
    for x, correct_digit in zip(X_train, y_train):
        
        # Create Target Output
        # Example: digit 3 -> [0 0 0 1 0 0 0 0 0 0]        
        y_target = np.zeros(10)
        y_target[correct_digit] = 1
        
        # Compute weighted sum
        z = W @ x + b
        
        # Compute output using step function
        y_output = np.where(z >= 0, 1, 0)

        # Compute error 
        delta = y_target - y_output
        
        # Update weights
        W = W + 0.001 * np.outer(delta, x) #Changed learning rate
        # np.outer calculates weighted change from the error (delta) times x
        
        # Update bias
        b = b + 0.001 * delta
        
        # Count classification errors
        if np.any(delta != 0):
            total_errors += 1
        
    # Store values for visual
    epoch_errors.append(total_errors)
    error_rate = round((100 -(total_errors/70000*100)),2)
    print(
        f"Epoch {epoch + 1}: "
        f"{total_errors} patterns with errors ("
        f"{error_rate} % accuracy)"
    )
    
    errors_by_rate[learning_rate] = epoch_errors
    
# TESTING

correct = 0 

for x, correct_digit in zip(X_test, y_test):
    
    z = W @ x + b
    # Choose neuron with largest weighted sum
    predicted_digit = np.argmax(z)
    y_pred_slp.append(predicted_digit)
    
#%%
'''
CONFUSION MATRICES
'''

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#CM for SLP
print("Confusion Matrix for SLP")
cm_slp = confusion_matrix(
    y_test,
    y_pred_slp,
    labels = np.arange(10)
    )

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_slp)
plt.title("Confusion Matrix SLP")
plt.show()
plt.tight_layout()

print(cm_slp)

# CM For MLP
print("Confusion Matrix for MLP")
y_pred = model.predict(X_test)


mlp_model = MLPClassifier(
    hidden_layer_sizes=128,
    max_iter=100,
    random_state=42, 
)

mlp_model.fit(X_train, y_train)

# Calculate numerical confusion matrix
cm_mlp = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(10)
)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix MLP")
plt.show()
plt.tight_layout()


# Figuring out most confused numbers overall and per model.

#%%
'''
DIGIT ACCURACY ANALYSIS
'''
overall_cm = cm_mlp + cm_slp

def top_3_digits(cm):
    total_incorrect = [] # Establishing an empty list for each cm 
    
    # Iterating over rows and sums of incorrect values to find most confused digits
    for i in np.arange(0,10):
        row = cm[i] 
        incorrect_sum = np.sum(row) - row[i] # Exlcuding diagonal values which are the index of row index
        
        total_incorrect.append(incorrect_sum) # appending with total incorrect values
    first = np.max(total_incorrect).astype(int) # The highest number of incorrect predictions
    first_index = total_incorrect.index(first) # The digit associated with the highest incorrect predictions number
    print(f"Most confused digit was {first_index}")
    print(f"It was predicted incorrectly {first} times")
    
    total_incorrect.remove(first) # Remove highest value to find second highest
    second = np.max(total_incorrect)
    
    if total_incorrect.index(second) > first_index:
        second_index = total_incorrect.index(second) + 1 # Adds 1 to offset index shift due to removal of previous max
    else: second_index = total_incorrect.index(second) # If second index is less than first, it won't be affected
        
    print(f"2nd Most confused digit was {second_index}")
    print(f"It was predicted incorrectly {second} times")
    
    
    total_incorrect.remove(second)
    third = np.max(total_incorrect)
    third_index = total_incorrect.index(third)
    
    if total_incorrect.index(third) > first_index:
        third_index  = total_incorrect.index(third) + 1
    elif (total_incorrect.index(third) > first_index and total_incorrect.index(third) > second_index):
        third_index  = total_incorrect.index(third) + 2
    else:third_index  = total_incorrect.index(third)
        
    
    print(f"3rd Most confused digit was {third_index}")
    print(f"It was predicted incorrectly {third} times")
    print()
    print()
    
matrices = {
    "overall_cm": overall_cm,
    "cm_mlp": cm_mlp,
    "cm_slp": cm_slp
}

for name, matrix in matrices.items():
    print(name)
    print('-'*40)
    top_3_digits(matrix)

#%%
'''
INDIVIDUAL PREDICTIONS
'''

# Finding incorrect images
total = 0
for i in np.arange(1000):
    if total < 50:
        if y_test[i] != y_pred[i]:
            plt.imshow(X_test[i].reshape(28, 28), cmap="grey_r")
            plt.title(f"Actual: {y_test[i]}  Predicted: {y_pred[i]}")
            plt.axis("off")
            plt.show()
            total += 1
            
# Correct Images
total = 0
for i in np.arange(1000):
    if total < 50:
        if y_test[i] == y_pred[i]:
            plt.imshow(X_test[i].reshape(28, 28), cmap="grey_r")
            plt.title(f"Actual: {y_test[i]}  Predicted: {y_pred[i]}")
            plt.axis("off")
            plt.show()
            total += 1
















