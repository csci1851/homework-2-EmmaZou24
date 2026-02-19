"""
Model stencil for Homework 2: Ensemble Methods with Gradient Boosting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, Union, List

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Set plotting style
sns.set_style("whitegrid")


class GradientBoostingModel:
    """Gradient Boosting model implementation with comprehensive evaluation and analysis tools"""

    def __init__(
        self,
        task: str = "classification",
        max_depth: int = 3,
        learning_rate: float = 0.1,
        n_estimators: int = 50,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: int = 42,
        use_scaler: bool = False,
    ):
        """
        Initialize Gradient Boosting model with customizable parameters

        Args:
            task: 'classification' or 'regression'
            max_depth: Maximum depth of a tree (controls pruning)
            learning_rate: Step size shrinkage to prevent overfitting
            n_estimators: Number of boosting rounds/trees
            subsample: Subsample ratio of training instances
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            max_features: Number of features to consider when looking for the best split
            random_state: Random seed for reproducibility
            use_scaler: Whether to apply StandardScaler before training/prediction
        """
        self.params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
        }

        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")

        self.model = None
        self.feature_names = None
        self.task = task
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None

    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Split data into training and testing sets

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test: Split datasets
        """
        # TODO: Implement train/test split and track feature names
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
        # pass

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = False):
        """
        Train the Gradient Boosting model

        Args:
            X_train: Training features
            y_train: Training targets
            verbose: Whether to print training progress

        Returns:
            self: Trained model instance
        """
        # TODO: Create classifier/regressor based on task and fit it
        if self.task == 'classification':
            self.model = GradientBoostingClassifier(learning_rate=self.params['learning_rate'], 
                                                    n_estimators=self.params['n_estimators'], 
                                                    max_depth=self.params['max_depth'],
                                                    subsample=self.params['subsample'],
                                                    min_samples_split=self.params['min_samples_split'],
                                                    min_samples_leaf=self.params['min_samples_leaf'],
                                                    max_features=self.params['max_features'],
                                                    random_state=self.params['random_state'],
                                                    verbose=verbose)
        else:
            self.model = GradientBoostingRegressor(learning_rate=self.params['learning_rate'], 
                                                    n_estimators=self.params['n_estimators'], 
                                                    max_depth=self.params['max_depth'],
                                                    subsample=self.params['subsample'],
                                                    min_samples_split=self.params['min_samples_split'],
                                                    min_samples_leaf=self.params['min_samples_leaf'],
                                                    max_features=self.params['max_features'],
                                                    random_state=self.params['random_state'],
                                                    verbose=verbose)
        if self.use_scaler:
            # X_train = self.scaler.fit_transform(X_train)
            X_col = X_train.columns
            X_index = X_train.index
            X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_col, index=X_index)
        self.model.fit(X_train, y_train)

    def predict(
        self, X: pd.DataFrame, return_proba: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions with the trained model

        Args:
            X: Feature data for prediction
            return_proba: If True and model is a classifier, return probability estimates

        Returns:
            Predictions or probability estimates
        """
        # TODO: Apply scaler when enabled, then predict
        if self.use_scaler:
            X_col = X.columns
            X_index = X.index
            X = pd.DataFrame(self.scaler.transform(X), columns=X_col, index=X_index)
        if return_proba:
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance on test data

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """

        # TODO: Compute metrics (classification vs regression)
        if self.task == "classification":
            metrics = {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
            }
            y_pred = self.predict(X_test)
            y_proba = self.predict(X_test, return_proba=True)
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            
            #binary vs multiclass
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                metrics['precision'] = precision_score(y_test, y_pred)
                metrics['recall'] = recall_score(y_test, y_pred)
                metrics['f1'] = f1_score(y_test, y_pred)
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        else:
            metrics = {"rmse": None, "mae": None, "r2": None}
            y_pred = self.predict(X_test)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> Dict:
        """
        Perform cross-validation

        Args:
            X: Feature data
            y: Target data
            cv: Number of cross-validation folds

        Returns:
            Dictionary of cross-validation results using sklearn cross_val_score
        """
        # TODO: Use Pipeline when scaling, and choose classifier/regressor based on task
        if self.task == 'classification':
            model = GradientBoostingClassifier(**self.params)
        else:
            model = GradientBoostingRegressor(**self.params)
        if self.use_scaler:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            # pipeline = Pipeline([
            #     ('model', model)
            # ])
            pipeline = model

        # TODO: Choose scoring metrics based on classification vs regression
        if self.task == "classification":
            scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        else:
            scoring = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]

        results = {score: cross_val_score(pipeline, X, y, scoring=score, cv=cv) for score in scoring}
        # TODO: Get mean, stdev of cross_val_score for each metric
        agg_results = dict()
        for score in scoring:
            res = results[score]
            agg_results[score] = []
            agg_results[score].append(np.mean(res))
            agg_results[score].append(np.std(res))
        return agg_results

    def get_feature_importance(
        self, plot: bool = False, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importances

        Returns:
            DataFrame with feature importances
        """
        try:
            features = self.model.feature_names_in_
        except:
            raise ValueError('model has not been fitted yet')
        imps = self.model.feature_importances_
        feature_tuples = [(features[i], imps[i]) for i in range(len(features))]
        feature_tuples.sort(key=lambda x: x[1], reverse=True)

        df = pd.DataFrame(feature_tuples, columns=['features', 'importance'])

        # TODO: Optionally plot a bar chart of top_n feature importances
        if plot:
            plt.bar(df['features'], df['importance'])
        return df


    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict,
        cv: int = 3,
        scoring: str = "roc_auc",
        plot: bool=False
    ) -> Dict:
        """
        Perform grid search for hyperparameter tuning

        Args:
            X: Feature data
            y: Target data
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            scoring: Scoring metric to evaluate

        Returns:
            Dictionary with best parameters and results
        """
        # TODO: Choose classifier or regressor based on task
        if self.task == 'classification':
            model = GradientBoostingClassifier(**self.params)
        else:
            model = GradientBoostingRegressor(**self.params)

        # TODO: Initialize GridSearchCV
        grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=cv)

        # TODO: Perform grid search for hyperparameter tuning
        grid_search.fit(X, y)
        
        if plot:
            return grid_search.cv_results_

        results = dict()
        results['best_params'] = grid_search.best_params_
        results['best_score'] = grid_search.best_score_
        return results

    def plot_tree(
        self, tree_index: int = 0, figsize: Tuple[int, int] = (20, 15)
    ) -> None:
        """
        Plot a specific tree from the ensemble

        Args:
            tree_index: Index of the tree to plot
            figsize: Figure size for the plot
        """

        pass


####################
### HW1 Classifier (modified to allow feature selection)
####################

class Hw1Classifier:
    def __init__(
        self,
        C: float = 1.0,
        random_state: int = 42,
        selected_features: Optional[List[str]] = None,
    ):
        """
        Initialize the classifier with specified parameters. Uses the lbfgs solver.

        Args:
            C: Inverse of regularization strength
            random_state: Random seed for reproducibility
        """
        self.C = C
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.selected_features = selected_features

    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features for the model.

        Returns:
            X_processed: Processed feature matrix
        """
        # TODO: Implement feature preprocessing
        # Default behavior should return X unchanged
        
        return X.copy()

    def fit(self, X: pd.DataFrame, y: pd.Series, scale: bool = True) -> None:
        """
        Preprocess features and fit the classification model.
        Save the fitted model in self.model.

        Args:
            X: Feature matrix
            y: Target variable (heart disease presence)
            scale: Whether to scale or not
        """
        # TODO: Implement model fitting
        # 1. Preprocess features via self.preprocess_features
        # 2. Scale features using self.scaler
        # 3. Initialize LogisticRegression
        # 4. Fit the model and store in self.model
        X_proc = self.preprocess_features(X)
        if self.selected_features:
            X_proc = X_proc.loc[:, self.selected_features]
        if scale:
            X_col = X_proc.columns
            X_index = X_proc.index
            X_proc = pd.DataFrame(self.scaler.fit_transform(X_proc), columns=X_col, index=X_index)
        self.model = LogisticRegression(C=self.C, max_iter=1000, random_state=self.random_state)
        self.model.fit(X_proc, y)

    def predict(self, X: pd.DataFrame, return_proba: bool = False, scale: bool = True) -> np.ndarray:
        """
        Make binary predictions using the trained model (self.model).

        Args:
            X: Feature matrix
            return_proba: If True, return probability of class 1 instead of hard labels

        Returns:
            y_pred: Binary predictions (0 or 1) if return_proba=False
            y_proba: Probability predictions for class 1 if return_proba=True
            scale: Whether to scale or not
        """
        # TODO: Implement prediction
        # 1. Ensure self.model is trained
        # 2. Preprocess features
        # 3. Scale using self.scaler.transform
        # 4. If return_proba: return self.model.predict_proba(X_scaled)[:, 1]
        #    else: return self.model.predict(X_scaled)
        if self.model is None:
            raise ValueError('Please fit your model first!')
        X_proc = self.preprocess_features(X)
        if self.selected_features:
            X_proc = X.loc[:, self.selected_features]
        if scale:
            X_col = X_proc.columns
            X_index = X_proc.index
            X_proc = pd.DataFrame(self.scaler.transform(X_proc), columns=X_col, index=X_index)
        if return_proba:
            return self.model.predict_proba(X_proc)[:,1]
        return self.model.predict(X_proc)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, scale: bool = True) -> Dict[str, float]:
        """
        Evaluate the model performance.

        Args:
            X: Feature matrix
            y: True target values

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # TODO: Implement model evaluation
        # Compute:
        #   y_pred  = self.predict(X)
        #   y_proba = self.predict(X, return_proba=True)
        #
        # Use zero_division=0 to avoid crashes on edge cases where a fold has no predicted positives:
        #   precision_score(y, y_pred, zero_division=0)
        #   recall_score(y, y_pred, zero_division=0)
        #   f1_score(y, y_pred, zero_division=0)
        #
        # ROC-AUC safeguard:
        #   Only compute roc_auc_score(y, y_proba) if BOTH classes appear in y (i.e., len(np.unique(y)) == 2).
        #   Otherwise, set "auc" to np.nan (or your chosen sentinel).
        #
        # Return dict with keys: "accuracy", "precision", "recall", "f1", "auc"
        if not self.model:
            raise ValueError('Please fit your model first!')
        #scaling
        # if scale:
        #     X = self.scaler.transform(X)
        y_pred = self.predict(X, scale=scale)
        y_proba = self.predict(X, return_proba=True, scale=scale)
        metrics = dict()
        metrics['accuracy'] = accuracy_score(y, y_pred)
        metrics['precision'] = precision_score(y, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y, y_pred, zero_division=0)
        metrics['auc'] = np.nan
        if len(np.unique(y))==2:
            metrics['auc'] = roc_auc_score(y, y_proba)
        return metrics

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, scale=True
    ) -> Dict[str, List[float]]:
        """
        Perform K-fold cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            n_splits: Number of folds for cross-validation

        Returns:
            cv_results: Dictionary with lists for each k-fold
        """
        # TODO: Implement K-fold cross-validation using KFold
        # For each fold:
        #   1. Split data into train/val
        #   2. Fit on train
        #   3. Evaluate on val
        #   4. Append metrics to cv_results
        #
        # Note: The ROC-AUC should effectively apply per fold:
        #   If y_val contains only one class, "auc" for that fold should be np.nan (or your chosen sentinel).
        folder = KFold(n_splits, shuffle=True, random_state=self.random_state)
        cv_results = dict()
        for train, test in folder.split(X, y):
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]
            self.fit(X_train, y_train, scale)
            metrics = self.evaluate(X_test, y_test, scale)
            for key in metrics.keys():
                if key not in cv_results:
                    cv_results[key] = []
                cv_results[key].append(metrics[key])
                    
        return cv_results
