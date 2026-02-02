import numpy as np
import seaborn as sns
import pandas as pd
from wcvpy.wcvp_download import get_all_taxa
from wcvpy.wcvp_name_matching import get_accepted_info_from_names_in_column

from data.get_knapsack_data import knapsack_plantae_compounds_csv
from data.get_wikidata import WCVP_VERSION, wikidata_plantae_compounds_csv

categories = ["AnimalFood", "EnvironmentalUses", "Fuels", "GeneSources", "HumanFood", "InvertebrateFood", "Materials", "Medicines", "Poisons",
              "SocialUses"]


def set_up_data():
    # Data from I. Ondo, IanOndo/UsefulPlants: UsefulPlants (v1.0.0), Zenodo, (2023); https://doi.org/10.5281/zenodo.8180352.
    # S. Pironon et al., ‘The Global Distribution of Plants Used by Humans’, Science 383, no. 6680 (2024): 293–97, https://doi.org/10.1126/science.adg8028.
    # plant_uses = pd.read_csv('inputs/UsefulPlants-v1.0.0/IanOndo-UsefulPlants-d7ef147/inst/extdata/utilised_plants_species_list.csv')
    # plant_uses['full_name'] = plant_uses['binomial_acc_name'] + ' ' + plant_uses['author_acc_name'].fillna('')
    # acc_plant_uses = get_accepted_info_from_names_in_column(plant_uses, 'full_name', wcvp_version=WCVP_VERSION)
    # acc_plant_uses.to_csv('outputs/_accepted_plant_use_data.csv')

    acc_plant_uses = pd.read_csv('outputs/_accepted_plant_use_data.csv', index_col=0)
    acc_plant_uses = acc_plant_uses.dropna(subset=['accepted_name'])

    # Deduplicate by taking max in categories for duplicate accepted_species
    final_acc_plant_uses = get_all_taxa(ranks=['Species'], accepted=True, version=WCVP_VERSION)[
        ['accepted_species', 'accepted_species_w_author']].set_index(
        'accepted_species')
    for cat in categories:
        cat_acc_plant_uses = acc_plant_uses.groupby('accepted_species').agg(cat=(cat, 'max'))
        cat_acc_plant_uses = cat_acc_plant_uses.rename(columns={'cat': cat})
        final_acc_plant_uses = pd.merge(final_acc_plant_uses, cat_acc_plant_uses[[cat]], left_index=True, right_index=True, how='left')
        final_acc_plant_uses[cat] = final_acc_plant_uses[cat].fillna(0)

    final_acc_plant_uses = final_acc_plant_uses.drop_duplicates()
    assert len(final_acc_plant_uses) == len(final_acc_plant_uses.index.unique().tolist())

    final_acc_plant_uses['Any Use'] = final_acc_plant_uses[categories].max(axis=1)

    wikidata = pd.read_csv(wikidata_plantae_compounds_csv, index_col=0)[['accepted_species']].dropna().drop_duplicates()
    knapsack = pd.read_csv(knapsack_plantae_compounds_csv, index_col=0)[['accepted_species']].dropna().drop_duplicates()

    all_phytochemical_data = pd.merge(wikidata, knapsack, on='accepted_species', how='outer')
    wikidata_species = wikidata['accepted_species'].unique().tolist()
    knapsack_species = knapsack['accepted_species'].unique().tolist()
    all_species = all_phytochemical_data['accepted_species'].unique().tolist()
    for sp in wikidata_species + knapsack_species:
        assert sp in all_species

    final_acc_plant_uses['In Phytochemical Data'] = final_acc_plant_uses.index.isin(all_phytochemical_data['accepted_species']).astype(int)

    final_acc_plant_uses = final_acc_plant_uses.sort_values(by='accepted_species_w_author')
    final_acc_plant_uses.to_csv('outputs/final_accepted_plant_use_data.csv')
    final_acc_plant_uses.describe(include='all').to_csv('outputs/final_accepted_plant_use_data_summary.csv')


def compare_to_phytochemical_data():
    final_use_data = pd.read_csv('outputs/final_accepted_plant_use_data.csv', index_col=0)

    from matplotlib import pyplot as plt

    labelled_traits = final_use_data[final_use_data['In Phytochemical Data'] == 1]

    all_traits = final_use_data[categories + ['Any Use']]
    labelled_traits = labelled_traits[categories + ['Any Use']]
    labelled_traits.describe(include='all').to_csv('outputs/labelled_accepted_plant_use_data_summary.csv')

    # Calculate means (proportions)
    labelled_means = labelled_traits.mean().reset_index()
    labelled_means.columns = ['Use', 'Proportion']
    labelled_means['Group'] = 'Species in Phytochemical Data'

    all_means = all_traits.mean().reset_index()
    all_means.columns = ['Use', 'Proportion']
    all_means['Group'] = 'All Species'

    # Combine for seaborn plotting
    plot_df = pd.concat([all_means, labelled_means], ignore_index=True)

    sns.set_theme()
    plt.figure(figsize=(8, 6))

    sns.barplot(
        data=plot_df,
        x='Use', y='Proportion', hue='Group',
        edgecolor='black'
    )
    plt.xticks(rotation=65)

    plt.legend(loc='upper left')
    plt.xlabel('Use')
    plt.ylabel('Proportion of Species with Use')
    plt.tight_layout()
    plt.savefig('outputs/use_plot.jpg', dpi=400)
    plt.close()
    plt.cla()
    plt.clf()


def main():
    # set_up_data()
    compare_to_phytochemical_data()


if __name__ == '__main__':
    main()
