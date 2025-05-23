'''
Training vs. Validation Datasets
When you split your images into 80% training and 20% validation, it generally means you're using the majority of your dataset to train the model (i.e., to adjust the weights and biases so the model learns patterns) and the remaining part to validate the model’s performance (i.e., to check if it’s learning generalizable patterns rather than just memorizing the training images) 

    - Training Dataset: 
        *Used for model fitting/learning by adjusting its internal parameters (weights).
        *The model sees these images repeatedly (in epochs), learns to recognize patterns, and minimizes the loss function accordingly.
    - Validation Dataset: 
        *Used to monitor how well the model is performing on previously unseen data during training.
        *Helps in detecting overfitting (when the model memorizes training images rather than learning generalizable features).
        *Guides in fine-tuning hyperparameters (like number of layers, learning rate, etc.) without biasing results toward the training set.
Once training is finished, you often use a totally separate test set (if available) to confirm the final accuracy and ensure the model generalizes well in the real world.


The general rule of thumb is 10000 images per label
'''