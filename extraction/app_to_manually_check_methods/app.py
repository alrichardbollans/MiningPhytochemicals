import os
import pickle

import pandas as pd
from shiny import App, reactive, render, ui
import sys

sys.path.append(os.path.join(os.environ.get('KEWSCRATCHPATH'), 'MiningPhytochemicals'))
from extraction.app_to_manually_check_methods.helper_functions import highlight_text

app_ui = ui.page_fluid(
    ui.row(
        ui.column(
            12,
            ui.h2("Upload Files to Begin"),
            ui.input_file("pkl_files", "Upload Annotation Files", multiple=True, accept=[".pkl"]),
            ui.input_file("previous_result_files", "Import Existing Results for pkls", multiple=False, accept=[".csv"]),
            ui.input_action_button("process_files", "Process Files", class_="btn btn-primary"),
            ui.hr()
        )
    ),
    ui.row(
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
        ui.column(
            4,
            ui.div(
                ui.h3("Details Panel"),
                ui.HTML(
                    f"Is compound: <b>{ui.output_text('compound_text')}</b> found in organism: "
                    f"<b><i>{ui.output_text('name_text')}</i></b> according to the text?"
                ),
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
                      "position: sticky; "
                      "top: 20px; "
                      "z-index: 1000; "
            ),
        ),
    )
)


def server(input, output, session):
    reactive_current_pkl_index = reactive.Value(0)
    reactive_current_taxon_index = reactive.Value(0)
    reactive_current_compound_index = reactive.Value(0)
    reactive_TaxaData_annotations = reactive.Value({})
    saved_results = reactive.Value([])
    reactive_finished = reactive.Value(False)
    reactive_previous_results = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.process_files)
    def handle_file_uploads():
        """Processes the uploaded files."""
        # Process Annotation Files
        pkl_files = input.pkl_files()
        uploaded_dicts = {}
        for pkl_file in pkl_files:
            with open(pkl_file['datapath'], "rb") as pkl_data:
                loaded_pkl = pickle.load(pkl_data)
                if len(loaded_pkl.taxa) == 0:
                    print(f'No taxa for {pkl_file}')
                    continue
                elif loaded_pkl.text is None:
                    print(f'No text for {pkl_file}')
                    continue
                else:
                    uploaded_dicts[pkl_file["name"]] = loaded_pkl

        reactive_TaxaData_annotations.set(uploaded_dicts)

        previous_result_files = input.previous_result_files()
        if previous_result_files is not None:
            print(previous_result_files)
            result_data = pd.read_csv(previous_result_files[0]['datapath'])
            reactive_previous_results.set(result_data)

    @output
    @render.ui
    def output_text():
        """Render highlighted text for the uploaded content."""
        annotations = reactive_TaxaData_annotations.get()
        if not annotations:
            return ui.p("No annotation files uploaded. Please upload annotation files to continue.")
        taxadata = list(annotations.values())[reactive_current_pkl_index.get()]

        text = taxadata.text

        taxon = taxadata.taxa[reactive_current_taxon_index.get()]

        highlighted = highlight_text(text, taxon.scientific_name,
                                     taxon.compounds[reactive_current_compound_index.get()])
        return ui.HTML(highlighted)

    def update_indices():
        """
        Update indices to move to the next compound, taxon, or taxadata object.
        """
        annotations = reactive_TaxaData_annotations.get()
        if not annotations:
            return

        # Get current objects
        taxadata_list = list(annotations.values())
        current_pkl_index = reactive_current_pkl_index.get()
        current_taxon_index = reactive_current_taxon_index.get()
        current_compound_index = reactive_current_compound_index.get()

        current_taxadata = taxadata_list[current_pkl_index]

        current_taxon = current_taxadata.taxa[current_taxon_index]

        # Move to the next compound
        if current_compound_index + 1 < len(current_taxon.compounds):
            reactive_current_compound_index.set(current_compound_index + 1)
        # No more compounds, move to the next taxon
        elif current_taxon_index + 1 < len(current_taxadata.taxa):
            reactive_current_compound_index.set(0)  # Reset compound index
            reactive_current_taxon_index.set(current_taxon_index + 1)
            # If next taxon dopesn't have compounds, move on
            if len(current_taxadata.taxa[current_taxon_index + 1].compounds) == 0:
                update_indices()
        # No more taxa, move to the next TaxaData object
        elif current_pkl_index + 1 < len(taxadata_list):
            reactive_current_compound_index.set(0)
            reactive_current_taxon_index.set(0)
            reactive_current_pkl_index.set(current_pkl_index + 1)
        else:
            # End of all data, do nothing
            print("No more items to process.")
            save_function()
            reactive_finished.set(True)
            return
        previous_results = reactive_previous_results.get()
        if previous_results is not None:
            current_pkl_index = reactive_current_pkl_index.get()
            current_taxon_index = reactive_current_taxon_index.get()
            current_compound_index = reactive_current_compound_index.get()

            current_taxadata = taxadata_list[current_pkl_index]
            current_taxon = current_taxadata.taxa[current_taxon_index]
            current_compound = current_taxon.compounds[current_compound_index]

            matching_row = previous_results[previous_results['taxon_name'] == current_taxon.scientific_name]
            matching_row = matching_row[matching_row['compound_name'] == current_compound]

            pkl_file = list(annotations.keys())[current_pkl_index]
            matching_row = matching_row[matching_row['pkl_file'] == pkl_file]
            if len(matching_row.index) > 0:
                update_indices()

    @output
    @render.text
    def name_text():
        """Render current name being displayed."""
        annotations = reactive_TaxaData_annotations.get()
        if annotations:
            taxadata = list(annotations.values())[reactive_current_pkl_index.get()]
            taxon = taxadata.taxa[reactive_current_taxon_index.get()]
            return taxon.scientific_name
        return "No annotation loaded."

    @output
    @render.text
    def compound_text():
        """Render current compound being highlighted."""
        annotations = reactive_TaxaData_annotations.get()
        if annotations:
            taxadata = list(annotations.values())[reactive_current_pkl_index.get()]

            taxon = taxadata.taxa[reactive_current_taxon_index.get()]

            return taxon.compounds[reactive_current_compound_index.get()]
        return "No annotation loaded."

    @reactive.Effect
    @reactive.event(input.submit_button)
    def on_yes_click():
        """
        Handle the user clicking "Yes".
        """
        if reactive_finished.get():
            return
        annotations = reactive_TaxaData_annotations.get()
        if not annotations:
            return
        pkl_file_name = list(annotations.keys())[reactive_current_pkl_index.get()]
        # Extract current context
        taxadata_list = list(annotations.values())
        current_pkl_index = reactive_current_pkl_index.get()
        current_taxon_index = reactive_current_taxon_index.get()
        current_compound_index = reactive_current_compound_index.get()

        current_taxadata = taxadata_list[current_pkl_index]
        current_taxon = current_taxadata.taxa[current_taxon_index]
        current_compound = current_taxon.compounds[current_compound_index]

        # Append result
        results = saved_results.get()
        results.append({
            "pkl_file": pkl_file_name,
            "taxon_name": current_taxon.scientific_name,
            "compound_name": current_compound,
            "decision": "Yes"
        })
        saved_results.set(results)  # Update reactive storage

        update_indices()

    @reactive.Effect
    @reactive.event(input.pass_button)
    def on_no_click():
        """
        Handle the user clicking "No".
        """
        if reactive_finished.get():
            return
        annotations = reactive_TaxaData_annotations.get()
        if not annotations:
            return
        pkl_file_name = list(annotations.keys())[reactive_current_pkl_index.get()]

        # Extract current context
        taxadata_list = list(annotations.values())
        current_pkl_index = reactive_current_pkl_index.get()
        current_taxon_index = reactive_current_taxon_index.get()
        current_compound_index = reactive_current_compound_index.get()

        current_taxadata = taxadata_list[current_pkl_index]
        current_taxon = current_taxadata.taxa[current_taxon_index]
        current_compound = current_taxon.compounds[current_compound_index]

        # Append result
        results = saved_results.get()
        results.append({
            "pkl_file": pkl_file_name,
            "taxon_name": current_taxon.scientific_name,
            "compound_name": current_compound,
            "decision": "No"
        })
        saved_results.set(results)  # Update reactive storage

        update_indices()

    def save_function():
        results = saved_results.get()

        if not results:
            print("No results to save.")
            return
        previous_results = reactive_previous_results.get()
        df = pd.DataFrame(results)
        output_path = os.path.join("outputs", "results.csv")
        pd.concat([previous_results, df]).to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")

    @reactive.Effect
    @reactive.event(input.save_button)
    def save_results():
        """
            Save all decisions to a CSV file.
            """
        save_function()


app = App(app_ui, server)
