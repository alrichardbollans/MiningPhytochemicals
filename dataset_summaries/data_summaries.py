import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from wcvpy.wcvp_download import get_all_taxa
import statsmodels.api as sm

from data.get_knapsack_data import knapsack_plantae_compounds_csv
from data.get_wikidata import wikidata_plantae_compounds_csv, WCVP_VERSION


def get_loess_outputs(data, x_var, y_var, outpath):
    # X = reg_data[[x_var]].values  # Independent variable
    X_to_plot = data[x_var].values  # Independent variable
    # scaled_data[metric] = np.log(scaled_data[metric])
    y = data[y_var].values  # Dependent variable

    # Fit LOESS with outlier robustness (iterations downweight outliers)
    loess_prediction = sm.nonparametric.lowess(exog=X_to_plot, endog=y, return_sorted=False)

    residuals = y - loess_prediction
    # Calculate R² (coefficient of determination)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R² (Coefficient of Determination): {r_squared:.3f}")

    data[f'{y_var}_loess_prediction'] = loess_prediction
    data[f'{y_var}_residuals'] = data[y_var] - data[f'{y_var}_loess_prediction']

    std_residual = data[f'{y_var}_residuals'].std()
    mean_residual = data[f'{y_var}_residuals'].mean()
    data[f'{y_var}_highlight_high'] = data[f'{y_var}_residuals'] > ((2 * std_residual) + mean_residual)
    data[f'{y_var}_highlight_low'] = data[f'{y_var}_residuals'] < (mean_residual - (2 * std_residual))
    data['R2'] = r_squared
    data.to_csv(os.path.join(outpath, f'{x_var}_and_{y_var}_loess_outputs.csv'))
    return data


def plot_2d_annotated_regression_data(data, x_var, y_var, outpath, extras_to_annotate: list = None):
    # Set up the plot
    import seaborn as sns

    data = get_loess_outputs(data, x_var, y_var, outpath)
    data['color'] = np.where((data[f'{y_var}_highlight_high'] == True), '#d12020', 'grey')
    data['color'] = np.where((data[f'{y_var}_highlight_low'] == True), '#5920ff', data['color'])

    sns.scatterplot(x=x_var, y=y_var, data=data, color=data['color'], edgecolor="black", alpha=0.8)

    # Highlight points where 'highlight' is True

    highlighted_data = data[(data[f'{y_var}_highlight_high'] == True) | (data[f'{y_var}_highlight_low'] == True)]
    to_annotate = highlighted_data[f'family'].unique().tolist()

    if extras_to_annotate is not None:
        to_annotate += extras_to_annotate
    for _, row in data.iterrows():
        if row['family'] in to_annotate:
            upshift = 0
            rightshift = -0.1
            plt.annotate(row['family'], (row[x_var] + rightshift, row[y_var] + upshift), ha='right', color='black')

    # Line plot for expected_diversity vs xvar

    ## Add estimator to smooth cases where multiple observations of the y variable at the same x
    sns.lineplot(x=x_var, y=f'{y_var}_loess_prediction', estimator='mean', color='black', linestyle='--', data=data)

    # Labels and legend

    plt.xlabel(x_var)

    plt.ylabel(y_var)

    plt.title('')

    # plt.legend(loc='upper right')

    plt.savefig(os.path.join(outpath, f'{x_var}_and_{y_var}_plot_with_outliers.jpg'), dpi=300)
    plt.close()
    plt.clf()
    sns.reset_orig()


def summarise(occurrences_csv: str, outpath):
    df = pd.read_csv(occurrences_csv, index_col=0)
    df.describe(include='all').to_csv(os.path.join(outpath, 'occurrences_summary.csv'))

    phytochemical_family_count = df.groupby('accepted_family')['accepted_species'].nunique()
    reg_data = pd.DataFrame(
        {'family': phytochemical_family_count.index, 'Species per Family in Data': phytochemical_family_count.values})
    families = get_all_taxa(version=WCVP_VERSION, accepted=True, ranks=['Species'])

    family_size = families.groupby('accepted_family')['accepted_species'].count()
    fam_df = pd.DataFrame({'family': family_size.index, 'Species per Family': family_size.values})

    reg_data = reg_data[reg_data['Species per Family in Data'] > 0]
    reg_data = pd.merge(reg_data, fam_df, on='family', how='left')

    plot_2d_annotated_regression_data(reg_data, 'Species per Family', 'Species per Family in Data', outpath)

def main():
    summarise(wikidata_plantae_compounds_csv, 'wikidata')
    summarise(knapsack_plantae_compounds_csv, 'knapsack')

if __name__ == '__main__':
    main()
