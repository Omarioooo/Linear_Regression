import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

class MultipleLinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.X = None
        self.y = None
        self.beta = None
        self.y_pred = None
        self.n_features = None
        self.n_sample = None
        self.residuals = None
        self.sst = None
        self.ssr = None
        self.sse = None
        self.mse = None
        self.msr = None
        self.r_squared = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_sample, self.n_features = self.X.shape
        
        # Add column of ones for intercept of B0
        self.X = np.c_[np.ones(self.n_sample), self.X]
        
        # Calculate coefficients using normal equation
        self.beta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y
        
        beta = self.beta
        self.intercept = beta[0]  # B0
        self.coefficients = beta[1:]  # B1, B2, B3, ....
        
        # Calculate the prediction and residuals
        self.y_pred = self.X @ beta
        self.residuals = self.y - self.y_pred
        
        # Calculate statistics
        y_bar = np.mean(self.y)
        self.sst = (self.y.T @ self.y) - self.n_sample * (y_bar ** 2)
        self.sse = (self.y.T @ self.y) - self.beta.T @ self.X.T @ self.y
        self.ssr = self.sst - self.sse
        
        df = self.n_sample - self.n_features - 1
        self.mse = self.sse / df
        
        df_res = self.n_features
        self.msr = self.ssr / df_res
        
        
        self.r_squared = self.ssr / self.sst

    def predict(self, X):
        X = np.array(X)
        
        # If input data doesn't B0, add it
        if X.shape[1] == self.n_features:
            X = np.c_[np.ones(X.shape[0]), X]
        
        beta = np.concatenate(([self.intercept], self.coefficients))
        return X @ beta

    def anova(self, alpha= 0.5):
        # Sum of squares
        sst = self.sst
        sse = self.sse
        ssr = self.ssr
        
        # Degrees of freedom
        df_total = self.n_sample - 1
        df_reg = self.n_features
        df_res = self.n_sample - self.n_features - 1
        
        # Mean Squares
        msr = self.msr
        mse = self.mse
        
        # F_statistic (F0)
        f_stat = msr / mse
        
        f_crit = stats.f.ppf(1 - alpha, df_reg, df_res)
        
        anova_results = {
            "SST": sst,
            "SSR": ssr,
            "SSE": sse,
            "DF_total": df_total,
            "DF_regression": df_reg,
            "DF_residual": df_res,
            "MSR": msr,
            "MSE": mse,
            "F_statistic": f_stat,
            "F_Critical" : f_crit,
            "Reject H0" : f_stat > f_crit
        }
        
        return anova_results

    def hypothesis_test(self, alpha=0.05):
        
        # Mean square error
        mse = self.mse
        
        # (XᵗX)^-1
        XtX_inv = np.linalg.inv(self.X.T @ self.X)
        
        # All coefficients (intercept + slopes)
        beta = self.beta
        
        # Standard errors of coefficients
        se = np.sqrt(np.diagonal(mse * XtX_inv))
        
        # t-statistics and p-values
        df_res = self.n_sample - self.n_features - 1
        t_stats = beta / se
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=df_res)) for t in t_stats]

        
        test_results = []
        for i in range(len(beta)):
            test_results.append(
            {
                "coefficient": f"B{i}",
                "coefficient_value": beta[i],
                "t_statistic": t_stats[i],
                "p_value": p_values[i],
                "significant": p_values[i] < alpha
            })
        
        return test_results

    def interval_estimation(self, alpha=0.05):
     
        # Calculate Mean Squared Error (MSE)
        mse = self.mse
        
        # Inverse of X^T * X
        XtX_inv = np.linalg.pinv(self.X.T @ self.X)
        
        # Get all coefficients
        beta = self.beta
        
        # Standard errors of coefficients
        se = np.sqrt(np.diagonal(mse * XtX_inv))
        
        # t-statistic critical value for alpha
        df_res = self.n_sample - self.n_features - 1
        t_critical = stats.t.ppf(1 - alpha / 2, df_res)
        
        # Confidence intervals
        confidence_intervals = []
        
        for i in range(len(beta)):
            margin_of_error = t_critical * se[i]
            lower_bound = beta[i] - margin_of_error
            upper_bound = beta[i] + margin_of_error
            
            confidence_intervals.append({
                "coefficient": f"B{i}",
                "coefficient_value": beta[i],
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            })
        
        return confidence_intervals
        
    def plot(self):
        # Remove intercept column (B0)
        X_features_only = self.X[:, 1:]
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_features_only)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create a mesh grid over the PCA components
        x_range = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 20)
        y_range = np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 20)
        xx, yy = np.meshgrid(x_range, y_range)

        # Regression coefficients (intercept + beta values)
        beta = self.beta
        # Compute the Z values (regression plane) using the linear equation
        zz = beta[0] + beta[1] * xx + beta[2] * yy  # B0 + B1*x + B2*y

        # Plot the regression plane
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='green', label='Regression Plane')

        # Scatter the actual data points (actual y-values)
        ax.scatter(X_pca[:, 0], X_pca[:, 1], self.y, color='blue', marker='o', label='Actual Data Points')

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Target Value')
        ax.set_title('Multiple Linear Regression 3D-Blot')
        ax.legend()

        plt.show()
