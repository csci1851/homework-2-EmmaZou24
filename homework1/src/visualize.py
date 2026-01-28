"""
Visualization utilities for Homework 1: Regression and Classification

This script provides visualization functions for:
1. Decision boundary visualization for logistic regression models
2. ROC-AUC curve visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any, Optional, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression


def plot_decision_boundary(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "coolwarm",
    alpha: float = 0.7,
    resolution: int = 100,
) -> plt.Axes:
    """
    Visualize the decision boundary of a fitted logistic regression model.

    This function reduces the dimensionality of the data to 2D using PCA
    and then plots the decision boundary of the model in this 2D space.
    The original model's coefficients are projected into the PCA space
    to maintain the same decision boundary characteristics.

    Args:
        model: Fitted logistic regression model
        X: Feature matrix
        y: Target labels (binary)
        feature_names: Names of features (optional)
        ax: Matplotlib axes object (optional)
        figsize: Figure size as (width, height)
        cmap: Colormap for the plot
        alpha: Transparency of scatter points
        resolution: Resolution of the decision boundary grid

    Returns:
        ax: Matplotlib axes with the plot
    """
    # Create axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Use feature names if provided, otherwise use column names or indices
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]

    # Convert to numpy array if DataFrame
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    y_array = y.values if isinstance(y, pd.Series) else y

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)

    # Reduce to 2 dimensions using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create a mesh grid for the decision boundary
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )

    # Project the original model's coefficients into the PCA space
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        # Get the coefficients and intercept from the original model
        original_coef = model.coef_[0]  # Shape: (n_features,)
        original_intercept = model.intercept_[0]

        # Project coefficients into PCA space
        # W_pca = W_orig * V.T where V is the PCA components matrix
        pca_coef = np.dot(original_coef, pca.components_)  # Shape: (2,)

        # The intercept remains the same
        pca_intercept = original_intercept

        # Calculate decision function values for the mesh grid
        Z = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                # Linear function: w0*x0 + w1*x1 + b
                point = np.array([xx[i, j], yy[i, j]])
                Z[i, j] = 1 / (1 + np.exp(-(np.dot(pca_coef, point) + pca_intercept)))
    else:
        # Fallback: Train a new model in PCA space if coefficients aren't available
        pca_model = LogisticRegression(
            C=getattr(model, "C", 1.0),
            random_state=getattr(model, "random_state", None),
            max_iter=1000,
        )
        pca_model.fit(X_pca, y_array)

        # Get the decision function values
        Z = pca_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    # Plot the data points
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], c=y_array, cmap=cmap, edgecolor="k", alpha=alpha
    )

    # Add colorbar
    plt.colorbar(contour, ax=ax)

    # Calculate and display PCA explained variance
    explained_variance = pca.explained_variance_ratio_
    explained_variance_sum = np.sum(explained_variance)

    # Add labels and title
    ax.set_xlabel(f"PCA Component 1 ({explained_variance[0]:.2%} variance)")
    ax.set_ylabel(f"PCA Component 2 ({explained_variance[1]:.2%} variance)")
    ax.set_title(
        f"Decision Boundary (Total variance explained: {explained_variance_sum:.2%})"
    )

    # Add legend
    handles, labels = scatter.legend_elements()
    ax.legend(handles, ["Class 0", "Class 1"], loc="upper right")

    # Add feature importance vectors
    if hasattr(model, "coef_"):
        # Transform feature importance to PCA space
        feature_importance = model.coef_[0]
        pca_components = pca.components_
        transformed_importance = np.dot(pca_components, feature_importance)

        # Scale for visualization
        scale = 5
        for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
            if abs(importance) > np.percentile(np.abs(feature_importance), 75):
                # Project feature onto PCA space
                feature_in_pca = np.dot(
                    pca_components, np.eye(len(feature_importance))[i]
                )
                ax.arrow(
                    0,
                    0,
                    feature_in_pca[0] * scale,
                    feature_in_pca[1] * scale,
                    head_width=0.1,
                    head_length=0.1,
                    fc="black",
                    ec="black",
                )
                ax.text(
                    feature_in_pca[0] * scale * 1.1,
                    feature_in_pca[1] * scale * 1.1,
                    name,
                    fontsize=9,
                )

    return ax
