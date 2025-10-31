import os
import pickle

import pandas as pd
from shiny import App, reactive, render, ui
import sys

sys.path.append(os.path.join(os.environ.get('KEWSCRATCHPATH'), 'MiningPhytochemicals'))
from data.get_data_with_full_texts import validation_data_csv
from extraction.app_to_manually_check_methods.helper_functions import highlight_text, \
    get_name_compound_pairs_for_doi, get_text

# Global results dictionary
result = {"found_pairs": [], "not_found_pairs": []}
doi_data_table = pd.read_csv(validation_data_csv, index_col=0)
dois = doi_data_table['refDOI'].unique().tolist()

found_cache_pkl = os.path.join('outputs', 'found_pairs.pkl')
try:
    found_cache = pickle.load(open(found_cache_pkl, 'rb'))
except FileNotFoundError:
    found_cache = []
    pickle.dump(found_cache, open(found_cache_pkl, 'wb'))
not_found_cache_pkl = os.path.join('outputs', 'not_found_pairs.pkl')

try:
    not_found_cache = pickle.load(open(not_found_cache_pkl, 'rb'))
except FileNotFoundError:
    not_found_cache = []
    pickle.dump(not_found_cache, open(not_found_cache_pkl, 'wb'))

# Define the Shiny app
app_ui = ui.page_fluid(
    ui.row(
        # Left column: Highlighted text display
        ui.column(
            8,
            ui.h2("Highlighted Text Viewer"),
            ui.output_ui("output_text", style=(
                "margin-top: 20px; "
                "padding: 20px; "
                "border: 1px solid #ccc; "
                "background-color: #fdfdfd; "
                "border-radius: 5px; "
                "box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); "
                "font-family: Arial, sans-serif; "
                "font-size: 16px; "
                "line-height: 1.6; "
                "overflow-y: auto; "
                "max-height: 400px;"
            )
                         )
        ),
        # Right column: Names in the name list/compound list and buttons
        ui.column(
            4,
            ui.div(
                ui.h3("Details Panel"),
                ui.HTML(
                    f"Is compound: <b>{ui.output_text("compound_text")}</b> found in organism: <b><i>{ui.output_text("name_text")}</i></b> according to the text?"),
                ui.p('Parts of possibly relevant organism names are highlighted in ',
                     ui.HTML('<span style="color: red; font-weight:bold;">Red</span>')),

                ui.p('Parts of possibly relevant compound names are highlighted in ',
                     ui.HTML('<span style="color: Blue; font-weight:bold;">Blue</span>')),
                ui.div(
                    ui.input_action_button("submit_button", "Yes", class_="btn btn-primary"),
                    ui.input_action_button("pass_button", "No", class_="btn btn-secondary",
                                           style="margin-left: 10px;"),
                    ui.input_action_button("save_button", "Save Results", class_="btn-success"),
                    style="margin-top: 20px;"
                ),
                style="background-color: #f8f9fa; padding: 10px; border: 1px solid #ccc;"
                      "position: sticky; "  # Sticky positioning
                      "top: 20px; "  # Space from the top of the viewport
                      "z-index: 1000; "
                ,

            ),
        ),
    )
)


def server(input, output, session):
    # Reactive values to store status of found/not found names
    # Reactive values for indices of the current DOI and pair
    reactive_current_pair_index = reactive.Value(0)
    reactive_current_doi_index = reactive.Value(0)
    reactive_current_model = reactive.Value('deepseek')

    # Reactive expression for the current text based on the DOI and pair index
    @reactive.Calc
    def current_text():
        current_doi_index = reactive_current_doi_index.get()
        doi = dois[current_doi_index]
        text = get_text(doi)
        pairs_for_doi = get_name_compound_pairs_for_doi(doi, reactive_current_model.get())
        if len(pairs_for_doi) == 0:
            skip()
            return current_text()
        else:
            name, compound = pairs_for_doi[reactive_current_pair_index.get()]
            if [name, compound, doi] in found_cache:

                result["found_pairs"].append((name, compound, doi))
                skip()
                return current_text()
            elif [name, compound, doi] in not_found_cache:
                result["not_found_pairs"].append((name, compound, doi))
                skip()
                return current_text()
            else:
                return doi, text, name, compound

    # Highlighted text output
    @output
    @render.ui
    def output_text():
        doi, text, name, compound = current_text()
        highlighted_text = highlight_text(
            text, name, compound
        )
        return ui.HTML(highlighted_text)

    # Name status panel
    @output
    @render.text
    def name_text():
        doi, text, name, compound = current_text()
        return name

    # Name status panel
    @output
    @render.text
    def compound_text():
        doi, text, name, compound = current_text()
        return compound

    @reactive.Calc
    def update_text():
        current_doi_index = reactive_current_doi_index.get()
        pairs_for_doi = get_name_compound_pairs_for_doi(dois[current_doi_index], reactive_current_model.get())
        # Move to the next pair
        new_pair_index = reactive_current_pair_index.get() + 1

        # Reset if we're beyond available pairs for the current DOI
        if new_pair_index >= len(pairs_for_doi):
            new_pair_index = 0
            new_doi_index = reactive_current_doi_index.get() + 1

            # Move to new model if we're at the end.
            if new_doi_index >= len(dois):
                if reactive_current_model.get() == 'deepseek':
                    save()
                    reactive_current_model.set('wikidata')
                    new_doi_index = 0
                    result["found_pairs"] = []
                    result["not_found_pairs"] = []
                else:
                    save()
                    raise Exception("No more DOIs or models to check.")
            print(f'Current doi: {dois[new_doi_index]}')
            reactive_current_doi_index.set(new_doi_index)

        reactive_current_pair_index.set(new_pair_index)

    @reactive.Calc
    def skip():
        update_text()

    # Handle Submit button click
    @reactive.Effect
    @reactive.event(input.submit_button)
    def submit_found_pair():
        doi, text, name, compound = current_text()

        # Update results

        result["found_pairs"].append((name, compound, doi))
        if [name, compound, doi] not in found_cache:
            found_cache.append([name, compound, doi])
            pickle.dump(found_cache, open(found_cache_pkl, 'wb'))
        update_text()

    @reactive.Effect
    @reactive.event(input.pass_button)
    def submit_not_found_pair():
        doi, text, name, compound = current_text()

        result["not_found_pairs"].append((name, compound, doi))
        if [name, compound, doi] not in not_found_cache:
            not_found_cache.append([name, compound, doi])
            pickle.dump(not_found_cache, open(not_found_cache_pkl, 'wb'))

        update_text()

    @reactive.Calc
    def save():
        print(f'Saving')
        df = pd.DataFrame(result["found_pairs"], columns=['name', 'compound', 'doi'])
        df.to_csv(os.path.join('outputs', reactive_current_model.get(), 'found_pairs.csv'), index=False)

        df = pd.DataFrame(result["not_found_pairs"], columns=['name', 'compound', 'doi'])
        df.to_csv(os.path.join('outputs', reactive_current_model.get(), 'not_found_pairs.csv'), index=False)

    @reactive.Effect
    @reactive.event(input.save_button)
    def save_results():
        save()

app = App(app_ui, server)
