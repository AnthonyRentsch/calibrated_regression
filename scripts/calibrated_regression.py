import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error

class CalibratedRegression:

    def __init__(self, X, y, model, cal_prop=0.2, cdf_method='bayesian', pp=None, pp_params=None):
        '''Initializes the class

        Parameters
        ----------
        X : np.array
            Data X
        y : np.array
            Data y
        model : pymc3 model or sklearn or statsmodels model
            The model to be calibrated
        cal_prop : float, optional, default: None
            The proportion of the training set to be used to make the calibration set
        cdf_method : string, optional, default: 'bayesian'
            Whether it is a Bayesian model, statsmodel or sklearn model
            Must be 'bayesian', 'bootstrap' or 'statsmodels'
        pp : function, optional, default: None
            The function to calculate the posterior predictive. Must return a numpy array.
        pp_params : dict, default: None
            Any additional parameters to be passed into the posterior predictive function.
        '''
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
        '''Fit underlying model

        Creates the calibration dataset and fits an IsotonicRegression on this dataset

        Returns
        -------
        self : CalibratedRegression object
        Returns a fit instance of the CalibratedRegression class
        '''

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
        '''Creates a Pandas dataframe which has the calibration dataset

        Parameters
        ----------
        X : np.array, optional, default: None
            Data X. Uses self.X_cal if None
        y : np.array, optional, default: None
            Data y. Uses self.y_cal if None
        pp : function, optional, default: None
            The function to calculate the posterior predictive. Must return a numpy array.
            Uses self.posterior_predictive if None
        pp_params : dict, default: None
            Any additional parameters to be passed into the posterior predictive function.
            Uses self.pp_params if None

        Returns
        -------
        calibration_dataset : Pandas dataframe
            THis contains X, y, predicted_cdf and empirical_cdf
        '''
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
        '''Return point estimates and PIs.

        Parameters
        ----------
        X_test : np.array
            Test data
        y_pred : np.array
            The predictions made by the model
        quantiles : list
            List of floats between 0 and 1 to be calibrated. Example: [0.05, 0.5, 0.95]

        Returns
        -------
        posterior_predictive_test : np.array
            Posterior predictive samples for the test data
        new_quantiles : list
            List of new floats, also between 0 and 1, that are the calibrated version of the input quantiles

        '''
        assert self.isotonic is not None, 'Call fit() first'
        new_quantiles = self.predict_quantiles(quantiles)

        # saving variables
        self.X_test = X_test
        self.y_pred = y_pred
        self.posterior_predictive_test = self.posterior_predictive(X_test, **self.pp_params)

        # returning quantiles
        return self.posterior_predictive_test, new_quantiles

    def predict_quantiles(self, quantiles):
        '''Returns transformed quantiles according to the isotonic regression model

        Parameters
        ----------
        quantiles : list
            List of floats between 0 and 1 to be calibrated. Example: [0.05, 0.5, 0.95]

        Returns
        -------
        quantiles_ : list
            List of new floats, also between 0 and 1, that are the calibrated version of the input quantiles

        '''
        assert self.isotonic is not None, 'Call fit() first'
        return self.isotonic.transform(quantiles)

    def pcdf(self, post_pred, y):
        '''Gets Predicted CDF

        Gets the predicted cdf, also represented as H(x_t)(y_t) in the paper.

        Parameters
        ----------
        post_pred : np.array
            Posterior predictive samples generated by the model (at a particular quantile).
        y : np.array
            The true data

        Returns
        -------
        pcdf_ : np.array
            The predicted cdf

        '''
        return np.mean(post_pred <= y.reshape(-1,1), axis=1)

    def ecdf(self, predicted_cdf):
        '''Empirical CDF.

        Gets the empirical cdf, also represented as $\hat{P}[H(x_t)(y_t)]$ in the paper.
        Counts how many points in the dataset have a pcdf <= to the pcdf of a point for all points in the dataset.

        Parameters
        ----------
        predicted_cdf : np.array
            Predicted cdf. Can be generated by calling self.pcdf for posterior predictive samples at a particular quantile.

        Returns
        -------
        ecdf_ : np.array
            The empirical cdf

        '''
        empirical_cdf = np.zeros(len(predicted_cdf))
        for i, p in enumerate(predicted_cdf):
            empirical_cdf[i] = np.sum(predicted_cdf <= p)/len(predicted_cdf)
        return empirical_cdf

    def plot_calibration_curve(self, ax):
        '''Plot calibration curve as described in paper (figure 3b).

        Parameters
        ----------
        ax : matplotlib axis object
            Axis to plot on

        Returns
        -------
        ax : matplotlib axis object
            Axis after it has been plotted on

        '''
        assert self.empirical_cdf is not None, 'Call fit() first'
        ax.scatter(self.predicted_cdf, self.empirical_cdf, alpha=0.7)
        ax.plot([0,1],[0,1],'--', color='grey', label='Perfect calibration')
        ax.set_xlabel('Predicted', fontsize=17)
        ax.set_ylabel('Empirical', fontsize=17)
        ax.set_title('Predicted CDF vs Empirical CDF', fontsize=17)
        ax.legend(fontsize=17)
        return ax

    def plot_diagnostic_curve(self, ax, X_test, y_test):
        '''Plot diagnostic curve as described in paper (figure 3c).

        Parameters
        ----------
        ax : matplotlib axis object
            Axis to plot on
        X_test : np.array
            Test data (X)
        y_test : np.array
            Test data (y). These are the predictions that need to be calibrated.

        Returns
        -------
        ax : matplotlib axis object
            Axis after it has been plotted on

        '''
        conf_level_lower_bounds = np.arange(start=0.025, stop=0.5, step=0.025)
        conf_levels = 1-2*conf_level_lower_bounds
        unc_pcts = []
        cal_pcts = []

        for cl_lower in conf_level_lower_bounds:
            quants = [cl_lower, 1-cl_lower]
            post_pred_test, new_quantiles = self.predict(X_test, y_test, quants)

            cal_lower, cal_upper = np.quantile(post_pred_test, new_quantiles, axis=1)
            unc_lower, unc_upper = np.quantile(post_pred_test, quants, axis=1)

            perc_within_unc = np.mean((y_test <= unc_upper)&(y_test >= unc_lower))
            perc_within_cal = np.mean((y_test <= cal_upper)&(y_test >= cal_lower))

            unc_pcts.append(perc_within_unc)
            cal_pcts.append(perc_within_cal)

        ax.plot([0,1],[0,1],'--', color='grey')
        ax.plot(conf_levels, unc_pcts, '-o', color='purple', label='uncalibrated')
        ax.plot(conf_levels, cal_pcts, '-o', color='red', label='calibrated')
        ax.legend(fontsize=14)
        ax.set_title('Diagnostic Plot', fontsize=17)
        ax.set_xlabel('Predicted Confidence Level', fontsize=17)
        ax.set_ylabel('Observed Confidence Level', fontsize=17)
        return ax

    def plot_intervals(self, ax, X_test, y_test, quantiles=[0.05, 0.5, 0.95]):
        '''Plot uncalibrated and calibrated predictive intervals.

        Parameters
        ----------
        ax : matplotlib axis object
            Axis to plot on
        X_test : np.array
            Test data (X)
        y_test : np.array
            Test data (y). These are the predictions that need to be calibrated.
        quantiles : list, optional, default=[0.05, 0.5, 0.95]
            List of floats between 0 and 1 to be calibrated.

        Returns
        -------
        ax : matplotlib axis object
            Axis after it has been plotted on

        '''
        assert len(ax)==2, 'Need to provide two axes'

        post_pred_test, new_quantiles = self.predict(X_test, y_test, quantiles)
        cal_lower, cal_median, cal_upper = np.quantile(post_pred_test, new_quantiles, axis=1)
        unc_lower, unc_median, unc_upper = np.quantile(post_pred_test, quantiles, axis=1)
        perc_within_unc = np.mean((y_test <= unc_upper)&(y_test >= unc_lower))
        perc_within_cal = np.mean((y_test <= cal_upper)&(y_test >= cal_lower))

        ax[0].plot(X_test, y_test, 'o', color='black', alpha=0.2, markersize=3)
<<<<<<< HEAD
        ax[0].set_title(f'Uncalibrated: {100*perc_within_unc:.2f}% of the test points within {round((1-2*quantiles[0])*100)}% interval', 
=======
        ax[0].set_title(f'Uncalibrated: {100*perc_within_unc:.2f}% of the test points within 90% interval',
>>>>>>> master
            fontsize=17)
        ax[0].set_xlabel('X', fontsize=17)
        ax[0].set_ylabel('y', fontsize=17)
        ax[0].fill_between(X_test, unc_lower, unc_upper, color='green', alpha=0.2)
        ax[0].plot(X_test, unc_median, label=f'Median. MSE={mean_squared_error(y_test, unc_median):.2f}')
        ax[0].legend(fontsize=17)

        ax[1].plot(X_test, y_test, 'o', color='black', alpha=0.2, markersize=3)
<<<<<<< HEAD
        ax[1].set_title(f'Calibrated: {100*perc_within_cal:.2f}% of the test points within {round((1-2*quantiles[0])*100)}% interval', 
=======
        ax[1].set_title(f'Calibrated: {100*perc_within_cal:.2f}% of the test points within 90% interval',
>>>>>>> master
            fontsize=17)
        ax[1].set_xlabel('X', fontsize=17)
        ax[1].set_ylabel('y', fontsize=17)
        ax[1].fill_between(X_test, cal_lower, cal_upper, color='yellow', alpha=0.2)
        ax[1].plot(X_test, cal_median, label=f'Median. MSE={mean_squared_error(y_test, cal_median):.2f}')
        ax[1].legend(fontsize=17)

        return ax, (cal_lower, cal_median, cal_upper), (unc_lower, unc_median, unc_upper)
