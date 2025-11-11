import os
import pathlib
import pickle

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
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


def get_regression_outputs(data, x_var, y_var, outpath):
    # X_to_plot = data[x_var].values  # Independent variable
    y = data[y_var].values  # Dependent variable

    model = LinearRegression()
    model.fit(data[[x_var]], y)
    reg_prediction = model.predict(data[[x_var]])

    residuals = y - reg_prediction
    # Calculate R² (coefficient of determination)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R² (Coefficient of Determination): {r_squared:.3f}")

    data[f'{y_var}_prediction'] = reg_prediction
    data[f'{y_var}_residuals'] = data[y_var] - data[f'{y_var}_prediction']

    std_residual = data[f'{y_var}_residuals'].std()
    mean_residual = data[f'{y_var}_residuals'].mean()
    data[f'{y_var}_highlight_high'] = data[f'{y_var}_residuals'] > ((2 * std_residual) + mean_residual)
    data[f'{y_var}_highlight_low'] = data[f'{y_var}_residuals'] < (mean_residual - (2 * std_residual))
    data['R2'] = r_squared
    data.to_csv(os.path.join(outpath, f'{x_var}_and_{y_var}_regression_outputs.csv'))
    return data


def plot_2d_annotated_regression_data(data, x_var, y_var, outpath, column_to_annotate: str,
                                      extras_to_annotate: list = None):
    # Set up the plot
    import seaborn as sns

    data = get_regression_outputs(data, x_var, y_var, outpath)

    if 'Region' in data.columns:
        plot_dist_of_metric(data, 'Species in Data_residuals',os.path.join(outpath, f'{x_var}_and_{y_var}_residuals_plot.jpg'))

    data['color'] = np.where((data[f'{y_var}_highlight_high'] == True), '#d12020', 'grey')
    data['color'] = np.where((data[f'{y_var}_highlight_low'] == True), '#5920ff', data['color'])

    sns.scatterplot(x=x_var, y=y_var, data=data, color=data['color'], edgecolor="black", alpha=0.8)

    # Highlight points where 'highlight' is True

    highlighted_data = data[(data[f'{y_var}_highlight_high'] == True) | (data[f'{y_var}_highlight_low'] == True)]
    to_annotate = highlighted_data[column_to_annotate].unique().tolist()

    if extras_to_annotate is not None:
        to_annotate += extras_to_annotate
    for _, row in data.iterrows():
        if row[column_to_annotate] in to_annotate:
            upshift = 0
            rightshift = -0.1
            plt.annotate(row[column_to_annotate], (row[x_var] + rightshift, row[y_var] + upshift), ha='right',
                         color='black')

    # Line plot for expected_diversity vs xvar

    ## Add estimator to smooth cases where multiple observations of the y variable at the same x
    sns.lineplot(x=x_var, y=f'{y_var}_prediction', estimator='mean', color='black', linestyle='--', data=data)

    # Labels and legend

    plt.xlabel(x_var)

    plt.ylabel(y_var)

    plt.title('')

    # plt.legend(loc='upper right')

    plt.savefig(os.path.join(outpath, f'{x_var}_and_{y_var}_plot_with_outliers.jpg'), dpi=300)
    plt.close()
    plt.clf()
    sns.reset_orig()


def plot_dist_of_metric(df_with_region_data, metric, out_path: str = None, colormap: str = 'inferno'):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader

    tdwg3_shp = shpreader.Reader(
        os.path.join('inputs', 'wgsrpd-master', 'level3', 'level3.shp'))
    tdwg3_region_codes = df_with_region_data['Region'].values

    ## Colour maps range is 0 - 1, so the values are standardised for this
    max_val = df_with_region_data[metric].max()
    min_val = df_with_region_data[metric].min()
    norm = plt.Normalize(min_val, max_val)
    print('plotting countries')

    plt.figure(figsize=(15, 9.375))
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linewidth=2)

    cmap = mpl.colormaps[colormap]
    for country in tdwg3_shp.records():

        tdwg_code = country.attributes['LEVEL3_COD']
        if tdwg_code in tdwg3_region_codes:
            ax.add_geometries([country.geometry], ccrs.PlateCarree(),
                              facecolor=cmap(
                                  norm(df_with_region_data.loc[df_with_region_data['Region'] == tdwg_code, metric].iloc[
                                           0])),
                              label=tdwg_code)

        else:
            ax.add_geometries([country.geometry], ccrs.PlateCarree(),
                              facecolor='white',
                              label=tdwg_code)

    all_map_isos = [country.attributes['LEVEL3_COD'] for country in tdwg3_shp.records()]
    missed_names = [x for x in tdwg3_region_codes if x not in all_map_isos]
    print(f'iso codes not plotted on map: {missed_names}')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    plt.tight_layout()
    fig = plt.gcf()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.175, 0.02, 0.65])
    cbar1 = fig.colorbar(sm, cax=cbar_ax)
    cbar1.ax.tick_params(labelsize=30)

    pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    plt.cla()
    plt.clf()


def summarise(df: pd.DataFrame, outpath, do_regression=True):
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    df['pairs'] = df['accepted_name'] + df['InChIKey_simp']
    df.describe(include='all').to_csv(os.path.join(outpath, 'occurrences_summary.csv'))

    if do_regression:
        output_geographic_plots(df, outpath)

        phytochemical_family_count = df.groupby('accepted_family')['accepted_species'].nunique()
        reg_data = pd.DataFrame(
            {'family': phytochemical_family_count.index,
             'Species per Family in Data': phytochemical_family_count.values})
        families = get_all_taxa(version=WCVP_VERSION, accepted=True, ranks=['Species'])

        family_size = families.groupby('accepted_family')['accepted_species'].count()
        fam_df = pd.DataFrame({'family': family_size.index, 'Species per Family': family_size.values})

        reg_data = reg_data[reg_data['Species per Family in Data'] > 0]
        reg_data = pd.merge(reg_data, fam_df, on='family', how='left')

        plot_2d_annotated_regression_data(reg_data, 'Species per Family', 'Species per Family in Data',
                                          outpath, 'family')


def get_deepseek_accepted_output_as_df(dois: list):
    deepseek_df = pd.DataFrame()
    for doi in dois:
        deepseek_output = pickle.load(open(os.path.join(deepseek_pkls_path, sanitise_doi(doi) + '.pkl'), 'rb'))
        df = convert_taxadata_to_accepted_dataframe(deepseek_output)
        deepseek_df = pd.concat([deepseek_df, df])
    return deepseek_df


def get_underlying_sp_distributions():
    df = get_all_taxa(version=WCVP_VERSION, accepted=True, ranks=['Species'])
    plot_native_number_accepted_taxa_in_regions(df, wcvp_accepted_columns['species'], 'underlying_distributions',
                                                'underlying_species_distributions.jpg', include_extinct=True,
                                                wcvp_version=WCVP_VERSION,
                                                colormap='inferno')


def output_geographic_plots(df, outpath: str):
    df = df.dropna(subset=['accepted_species']).drop_duplicates(subset=['accepted_species'])
    df_with_dists = get_distributions_for_accepted_taxa(
        df, 'accepted_species', include_extinct=True,
        wcvp_version=WCVP_VERSION)[['accepted_species', 'native_tdwg3_codes']]

    df_with_dists = df_with_dists.explode('native_tdwg3_codes')
    count_df = df_with_dists.native_tdwg3_codes.value_counts().reset_index().rename(
        columns={'native_tdwg3_codes': 'Region', 'count': 'Species in Data'})

    underlying_species_region_counts = pd.read_csv(
        os.path.join('underlying_distributions', 'underlying_species_distributions.jpg_regions.csv'), index_col=0)
    underlying_species_region_counts = underlying_species_region_counts.rename(
        columns={'Number of Taxa': 'Species in Underlying Population'})

    analysis_df = pd.merge(count_df, underlying_species_region_counts, on='Region', how='left')
    analysis_df = analysis_df.sort_values(by='Species in Underlying Population')

    plot_2d_annotated_regression_data(analysis_df, 'Species in Underlying Population', 'Species in Data',
                                      outpath,
                                      'Region')


def main():
    # get_underlying_sp_distributions()
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
