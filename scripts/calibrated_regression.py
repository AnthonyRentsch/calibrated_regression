import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

class CalibratedRegression:
    
    def __init__(self, X, y, model, cal_prop=0.2, cdf_method='bayesian', pp=None, pp_params=None):
        '''Initializes the class'''
        # data
        self.X = X
        self.y = y
        
        # model
        self.model = model
        self.posterior_predictive = pp
        self.pp_params = pp_params
        
        # calibration features
        self.calibration_dataset = None
        self.isotonic = None
        
        # split up training and calibration sets
        self.X_train, self.X_cal, self.y_train, self.y_cal = train_test_split(X, y, test_size=cal_prop)
        
        if cdf_method in ['bayesian','bootstrap','statsmodels']:
            self.cdf_method = cdf_method
        else:
            raise ValueError("cdf_method must be of type 'bayesian', 'bootstrap', or 'statsmodels'")
    
    def bootstrap():
        '''Utility function to bootstrap.'''
        pass
    
    def fit(self):
        '''Fit underlying model'''
        
        if self.cdf_method == 'bayesian':
            # there should be a posterior_predictive function
            assert self.posterior_predictive is not None and self.pp_params is not None
            # call the posterior predictive function
            self.posterior_predictive_cal = self.posterior_predictive(self.X_cal, **self.pp_params)
            
        elif self.cdf_method == 'bootstrap':
            # get CDF from bootstrapping
            pass
        
        elif self.cdf_method == 'statsmodels':
            # get CDF from statsmodels 
            pass
        
        # create the calibration dataset
        self.calibration_dataset, self.predicted_cdf, self.empirical_cdf = self.create_calibration_dataset()
            
        # fit the isotonic regression
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        self.isotonic.fit(self.empirical_cdf, self.predicted_cdf)
            
        return self
        
    def create_calibration_dataset(self, X=None, y=None, pp=None, pp_params=None):
        '''Creates a Pandas dataframe which has the calibration dataset''' 
        # check conditions
        X = X if X is not None else self.X_cal
        y = y if y is not None else self.y_cal
        pp = pp if pp is not None else self.posterior_predictive
        pp_params = pp_params if pp_params is not None else self.pp_params
        
        post_pred = pp(X, **pp_params)
        predicted_cdf = self.pcdf(post_pred, y) # predicted CDF
        empirical_cdf = self.ecdf(predicted_cdf) # empirical CDF
        
        # putting results in a Pandas dataframe
        calibration_dataset = pd.DataFrame({'X': X, 'y': y, 
                                            'predicted_cdf': predicted_cdf, 
                                            'empirical_cdf': empirical_cdf})
        
        return calibration_dataset[['X','y','predicted_cdf','empirical_cdf']], predicted_cdf, empirical_cdf
    
    def predict(self, X_test, y_pred, quantiles):
        '''Return point estimates and PIs.'''
        assert self.isotonic is not None, 'Call fit() first'
        new_quantiles = self.predict_quantiles(quantiles)
        
        # saving variables
        self.X_test = X_test
        self.y_pred = y_pred
        self.posterior_predictive_test = self.posterior_predictive(X_test, **self.pp_params)
        
        # returning quantiles
        return self.posterior_predictive_test, new_quantiles
    
    def predict_quantiles(self, quantiles):
        '''Returns transformed quantiles according to the isotonic regression model'''
        assert self.isotonic is not None, 'Call fit() first'
        return self.isotonic.transform(quantiles)
    
    def pcdf(self, post_pred, y):
        '''Gets Predicted CDF'''
        return np.mean(post_pred <= y.reshape(-1,1), axis=1)
        
    def ecdf(self, predicted_cdf):
        '''Empirical CDF.'''
        empirical_cdf = np.zeros(len(predicted_cdf))
        for i, p in enumerate(predicted_cdf):
            empirical_cdf[i] = np.sum(predicted_cdf <= p)/len(predicted_cdf)
        return empirical_cdf
    
    def plot_calibration_curve(self, ax):
        '''Plot calibration curve as described in paper (figure 3b).'''
        assert self.empirical_cdf is not None, 'Call fit() first'
        ax.scatter(self.predicted_cdf, self.empirical_cdf, alpha=0.7)
        ax.plot([0,1],[0,1],'--', color='grey')
        ax.set_xlabel('Predicted', fontsize=17)
        ax.set_ylabel('Empirical', fontsize=17)
        ax.set_title('Predicted CDF vs Empirical CDF')
        return ax
    
    def plot_diagnostic_curve(self, ax, intervals):
        '''Plot diagnostic curve as described in paper (figure 3c).'''
        assert self.posterior_predictive_test.shape, 'Call predict() first'
        
        # uncalibrated
        observed_uncalibrated = np.mean(self.predicted_cdf.reshape(1, -1) <= intervals, axis=1) 
        ax.plot(intervals, observed_uncalibrated, 'o-', color='purple', label='uncalibrated')
        
        # calibrated
        predicted_values = self.pcdf(self.posterior_predictive_test, self.y_pred)
        calibrated_values = self.isotonic.predict(predicted_values)
        observed_calibrated = np.mean(calibrated_values.reshape(1, -1) <= intervals, axis=1) 
        ax.plot(intervals, observed_calibrated, 'o-', color='red', label='calibrated')
        
        # default line
        ax.plot([0,1],[0,1], '--', color='black', alpha=0.7)
        ax.set_ylabel('Observed Confidence Level', fontsize=17)
        ax.set_xlabel('Expected Confidence Level', fontsize=17)
        ax.legend(fontsize=17)
        return ax