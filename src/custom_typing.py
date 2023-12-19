from pydantic import BaseModel, Field
from typing import Optional, List, Union


class Data(BaseModel):
    features: List[List[float]] = Field(
        description="Data features",
        examples=[[[1.1, 2.4, 3.2], [3.1, 2.3, 4.5]]]
    )
    targets: Optional[List[int]] = Field(
        description="Target values, optional for prediction",
        examples=[[1, 4]]
    )


class LinearRegressionConfig(BaseModel):
    fit_intercept: Optional[bool] = Field(
        description='Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.'
    )


class DecisionTreeRegressorConfig(BaseModel):
    max_depth: Optional[int] = Field(description="The maximum depth of the tree.")
    min_samples_split: Optional[Union[int, float]] = Field(
        description="The minimum number of samples required to split an internal node."
    )
    min_samples_leaf: Optional[Union[int, float]] = Field(
        description="The minimum number of samples required to be at a leaf node."
    )
    min_weight_fraction_leaf: Optional[float] = Field(
        description="The minimum weighted fraction of the sum total of weights \
            (of all the input samples) required to be at a leaf node."
    )
    max_features: Optional[Union[int, float]] = Field(
        description="The number of features to consider when looking for the best split"
    )
    max_leaf_nodes: Optional[int] = Field(
        description="Grow a tree with ``max_leaf_nodes`` in best-first fashion."
    )
    min_impurity_decrease: Optional[float] = Field(
        description="A node will be split if this split induces a decrease \
            of the impurity greater than or equal to this value."
    )


class RandomForestRegressorConfig(DecisionTreeRegressorConfig):
    n_estimators: Optional[int] = Field(description="The number of trees in the forest.")
    bootstrap: Optional[bool] = Field(
        description="Whether bootstrap samples are used when building trees. \
            If False, the whole dataset is used to build each tree."
    )
    max_samples: Optional[Union[int, float]] = Field(
        description="If bootstrap is True, the number of samples \
            to draw from X to train each base estimator.")


class SVRConfig(BaseModel):
    kernel: Optional[str] = Field(description="Specifies the kernel type to be used in the algorithm.")
    degree: Optional[int] = Field(description="Degree of the polynomial kernel function ('poly').")
    gamma: Optional[Union[str, float]] = Field(description="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.")
    coef0: Optional[float] = Field(description="Independent term in kernel function.")
    tol: Optional[float] = Field(description="Tolerance for stopping criterion.")
    C: Optional[float] = Field(description="Regularization parameter.")
    epsilon: Optional[float] = Field(description="Epsilon in the epsilon-SVR model.")
    shrinking: Optional[bool] = Field(description="Whether to use the shrinking heuristic.")
    max_iter: Optional[int] = Field(description="Hard limit on iterations within solver, or -1 for no limit.")
