import os
import pathlib
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from wcvpy.wcvp_download import get_all_taxa, wcvp_accepted_columns, get_distributions_for_accepted_taxa, \
    plot_native_number_accepted_taxa_in_regions
import statsmodels.api as sm

from data.get_data_with_full_texts import validation_data_csv
from data.get_knapsack_data import knapsack_plantae_compounds_csv
from data.get_papers_with_no_hits import get_sanitised_dois_for_papers
from data.get_wikidata import wikidata_plantae_compounds_csv, WCVP_VERSION
from data.parse_refs import sanitise_doi
from extraction.methods.get_agreements_and_disagreements import convert_taxadata_to_accepted_dataframe
from extraction.methods.running_models import deepseek_pkls_path


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


def summarise(df: pd.DataFrame, outpath, do_regression=True):
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    df['pairs'] = df['accepted_name'] + df['InChIKey_simp']
    df.describe(include='all').to_csv(os.path.join(outpath, 'occurrences_summary.csv'))

    output_geographic_plots(df, outpath)

    if do_regression:
        phytochemical_family_count = df.groupby('accepted_family')['accepted_species'].nunique()
        reg_data = pd.DataFrame(
            {'family': phytochemical_family_count.index,
             'Species per Family in Data': phytochemical_family_count.values})
        families = get_all_taxa(version=WCVP_VERSION, accepted=True, ranks=['Species'])

        family_size = families.groupby('accepted_family')['accepted_species'].count()
        fam_df = pd.DataFrame({'family': family_size.index, 'Species per Family': family_size.values})

        reg_data = reg_data[reg_data['Species per Family in Data'] > 0]
        reg_data = pd.merge(reg_data, fam_df, on='family', how='left')

        plot_2d_annotated_regression_data(reg_data, 'Species per Family', 'Species per Family in Data', outpath)


def get_deepseek_accepted_output_as_df(dois: list):
    deepseek_df = pd.DataFrame()
    for doi in dois:
        deepseek_output = pickle.load(open(os.path.join(deepseek_pkls_path, sanitise_doi(doi) + '.pkl'), 'rb'))
        df = convert_taxadata_to_accepted_dataframe(deepseek_output)
        deepseek_df = pd.concat([deepseek_df, df])
    return deepseek_df


def output_geographic_plots(df, outpath: str):
    df = df.dropna(subset=['accepted_species'])

    plot_native_number_accepted_taxa_in_regions(df, wcvp_accepted_columns['species'], outpath,
                                                'underlying_species_distributions.jpg', include_extinct=True,
                                                wcvp_version=WCVP_VERSION,
                                                colormap='inferno')
    underlying_species_region_counts = pd.read_csv(
        os.path.join(outpath, 'underlying_species_distributions.jpg_regions.csv'), index_col=0)
    species_regions = pd.read_csv()
    analysis_df = pd.merge(df, species_regions, on='accepted_species', how='left')

    analysis_df = analysis_df.explode('native_tdwg3_codes')
    # count_df = working_data.native_tdwg3_codes.value_counts().reset_index().rename(
    #     columns={'index': 'Region', 0: 'count'})
    #

    #
    #
    # analysis_df = analysis_df.sort_values(by=x_var)


def main():
    summarise(pd.read_csv(wikidata_plantae_compounds_csv, index_col=0), 'wikidata')
    summarise(pd.read_csv(knapsack_plantae_compounds_csv, index_col=0), 'knapsack')
    doi_data_table = pd.read_csv(validation_data_csv, index_col=0)
    dois = doi_data_table['refDOI'].unique().tolist()
    deepseek_df = get_deepseek_accepted_output_as_df(dois)
    summarise(deepseek_df, 'deepseek_validaton', do_regression=False)

    phytochem_txt_dir, result = get_sanitised_dois_for_papers('phytochemistry papers')
    summarise(get_deepseek_accepted_output_as_df(result), 'deepseek_phytochem_papers', do_regression=False)


if __name__ == '__main__':
    main()
