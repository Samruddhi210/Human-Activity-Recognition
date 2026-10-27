# Human-Activity-Recognition

Explainable Human Activity Recognition Using CNN–LSTM:

An AI-powered video classification project that combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to recognize human activities from video sequences. The project focuses on learning spatial features from video frames, modeling temporal patterns across frames, and exploring explainability to make model predictions easier to interpret.

Overview

Human Activity Recognition (HAR) is a computer vision task that involves identifying human actions from visual or sensor-based data. Video-based HAR is challenging because activities can differ in subtle movements, duration, posture, and temporal patterns.
This project uses a hybrid CNN–LSTM deep-learning architecture to analyze video sequences and classify human activities into predefined categories.
The CNN component extracts spatial features from individual video frames, while the LSTM component learns temporal dependencies across the sequence. This combination allows the model to consider both visual appearance and movement over time.
The project also explores explainable AI (XAI) as a way to investigate which visual regions contribute to model predictions and to analyze potential misclassifications.

Objectives

Develop a deep-learning pipeline for video-based human activity recognition.

Extract spatial features from video frames using a CNN.

Model sequential and temporal information using an LSTM.

Train and evaluate a multiclass activity-classification model.

Analyze model performance using suitable classification metrics.

Explore visual explainability techniques to understand model behavior.

Investigate misclassified examples to identify potential weaknesses and opportunities for improvement.

Key Features:

Video-based activity classification: Predict an activity category from a video sequence.

Spatial feature extraction: Use convolutional layers to learn visual patterns from frames.

Temporal sequence modeling: Use an LSTM to learn dependencies across frame-level features.

Deep-learning pipeline: Support video preprocessing, model training, evaluation, and inference.

Performance analysis: Evaluate classification performance using appropriate metrics.

Explainability-oriented analysis: Explore visual explanations and examine examples where predictions are incorrect.

Note: Features should be marked as implemented only after they have been developed and tested.

Model Architecture: 

The proposed architecture follows a CNN–LSTM sequence-classification approach.

1. Video preprocessing

Video data is converted into a sequence of frames suitable for model input. Preprocessing may include:

Reading video files.

Sampling frames from each video.

Resizing frames to a consistent spatial resolution.

Converting frames into numerical arrays or tensors.

Normalizing pixel values.

Associating each sequence with its corresponding activity label.

The exact frame-sampling strategy, input dimensions, and normalization method depend on the dataset and implementation.

2. CNN-based spatial feature extraction:

A Convolutional Neural Network processes individual frames to learn visual representations.

CNN layers can learn patterns such as edges, shapes, body configurations, and higher-level visual features. These frame-level representations are then used as inputs to the temporal model.

For a sequence containing multiple frames, the CNN extracts a feature vector from each frame while maintaining the temporal order of the sequence.

3. LSTM-based temporal modeling:

Long Short-Term Memory networks are designed to model sequential dependencies. In this project, the LSTM receives the sequence of CNN-extracted frame features.

It learns temporal relationships that can help distinguish activities with similar appearances but different movement patterns.

The resulting sequence representation is passed to a classification layer.

4. Activity classification:

A fully connected classification layer maps the learned representation to the available activity classes. For multiclass classification, a softmax operation can convert the output logits into class probabilities.

The predicted activity is the class with the highest output score.

Architecture summary

Video sequence → Frame preprocessing → CNN feature extraction → LSTM temporal modeling → Classification layer → Predicted activity

Explainable AI (XAI)

A key research direction for this project is understanding why a model produces a particular activity prediction rather than treating its output as a black box.

Potential explainability methods include:

Grad-CAM: Generate class-specific visual heatmaps to inspect image regions influencing a CNN prediction.

Frame-level analysis: Compare explanations across different frames to investigate whether relevant visual information changes over time.

Misclassification analysis: Examine incorrectly classified videos and identify patterns such as ambiguous movements, occlusion, poor lighting, or background variation.

Prediction confidence analysis: Compare confidence scores across correct and incorrect predictions.

For a CNN–LSTM architecture, Grad-CAM must be applied to an appropriate convolutional feature layer and connected to the target class score. Frame-level heatmaps can then be examined alongside the temporal sequence.

Explainability methods provide diagnostic evidence about model behavior; they do not automatically establish causality or guarantee that a prediction is correct.

Technologies Used:

Python: Primary programming language.

Deep Learning: CNN–LSTM architecture for spatial and temporal feature learning.

PyTorch or TensorFlow/Keras: Deep-learning framework, depending on the implementation.

OpenCV: Video reading, frame extraction, and image preprocessing.

NumPy: Numerical operations and array manipulation.

Matplotlib: Visualization of training history, evaluation metrics, and explanations.

Scikit-learn: Classification metrics and evaluation utilities, if used.

Grad-CAM: Visual explanation technique, if implemented.

Jupyter Notebook / Google Colab: Development and experimentation environment

 Dataset — UCF101 Action Recognition: 

This project uses the **UCF101 Action Recognition Dataset**, a widely used benchmark for human activity recognition from video.

- Dataset:  UCF101 — Action Recognition
- Original dataset:  [University of Central Florida — UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/)
- Kaggle source:  [UCF101 Action Recognition Dataset](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition)
- Number of video clips: 13,320
- Number of activity classes: 101
- Data type: Video sequences
- Task: Multiclass human activity classification

Dataset characteristics:

The dataset includes activities spanning sports, body movements, human–object interactions, human–human interactions, and musical instrument performance.

Examples of activity classes include:

- Basketball Shooting
- Biking
- Playing Guitar
- Jumping Jack
- Walking with a Dog
- Typing
- Tennis Swing
- Playing Piano

UCF101 contains realistic videos collected from YouTube. Differences in camera motion, subject appearance, viewpoint, background clutter, and illumination make it useful for studying the generalization capabilities of video-classification models.

Dataset preprocessing:

For a CNN–LSTM pipeline, video preprocessing can include:

1. Loading each video and extracting frames.
2. Sampling a fixed number of frames per video.
3. Resizing frames to the input dimensions expected by the CNN.
4. Normalizing pixel values.
5. Preserving the temporal order of sampled frames.
6. Mapping each video to its corresponding activity label.

The actual frame count, image dimensions, normalization strategy, and sampling method should be documented according to the implementation used in this repository.

Train/test split considerations:

UCF101 provides established train/test splits for action-recognition experiments. Where feasible, use the official split files and document which split is used.

The original dataset groups related videos that may share backgrounds, viewpoints, or recording characteristics. Keeping related videos separated between training and testing is important because leakage can inflate performance estimates. Report the split protocol alongside evaluation results.

 Dataset availability:

The video dataset is hosted externally and should be downloaded from the source rather than committed directly to this GitHub repository. Follow the dataset provider's terms and include instructions for placing the files in the expected directory structure.

