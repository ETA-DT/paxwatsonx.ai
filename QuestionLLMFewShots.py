#!/usr/bin/env python
# coding: utf-8
import sys
import re
import requests
import os
import getpass
# from ibm_cloud_sdk_core import IAMTokenManager
# from ibm_cloud_sdk_core.authenticators import IAMAuthenticator, BearerTokenAuthenticator
from TM1py.Services import TM1Service
from TM1py.Utils.Utils import build_pandas_dataframe_from_cellset
from TM1py.Utils.Utils import build_cellset_from_pandas_dataframe
import random
from ibm_watsonx_ai.foundation_models import Model

class prompt:
    def _init_(self,access_token, project_id):
        self.access_token = access_token 
        self. project_id = project_id 

    def generate(self, input, model_id, parameters):
        wml_url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2024-05-23"
        Headers = {
            "Authorization": "Bearer " + self.access_token, 
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        data = {
            "model id" : model_id,
            "input": input,
            "parameters" : parameters,
            "project_id": self.project_id
        }
        response = requests.post(wml_url, json=data, headers=Headers)
        if response.status_code == 200:
            return response.json()["results"][0]["generated_text"] 
        else:
            return response.text  

# ## watsonx API connection
# This cell defines the credentials required to work with watsonx API for Foundation


# tm1 tango_core_model
tm1_user = "erwan.tang"         # getpass.getpass('Please enter your TM1 username: ')
tm1_password = "Datatilt2021"               # getpass.getpass('Please enter your TM1 password: ')

tm1_credentials = {
    "auth_type":"""basic""",
    "password" : tm1_password,
    "service_root": "91.236.254.119",
    "username" : tm1_user,
    "port" : 5029,
    "view_name" : "ZViewSource"    
}

# access_token = IAMTokenManager(
#     apikey = getpass.getpass("Please enter you api key (hit enter): "),
#     url = "https://iam.cloud.ibm.com/identity/token"
#     ).get_token()

def get_credentials():
    return {
        "url" : "https://us-south.ml.cloud.ibm.com",
        "apikey" : "2dq6RfeU0FiAPPJSShktlqGHd_mPYWlpd3Vc-brcEENG" # getpass.getpass("Please enter your api key (hit enter): ")
    }


# # Inferencing
# This cell demonstrated how we can use the model object as well as the created access token
# to pair it with parameters and input string to obtain
# the response from the the selected foundation model.
# 
# ## Defining the model id
# We need to specify model id that will be used for inferencing:

model_id = "meta-llama/llama-3-70b-instruct"

# ## Defining the model parameters
# We need to provide a set of model parameters that will influence the
# result:

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 800,
    "repetition_penalty": 1.2,
    "stop_sequences": ["'\'''\'''\''","\"\"\"","```"]
}


# ## Defining the project id or space id
# The API requires project id or space id that provides the context for the call. We will obtain
# the id from the project or space in which this notebook runs:

project_id = "900b47ad-2347-4b05-a45a-ccd8ce2898d4"     # os.getenv("PROJECT_ID")
space_id = "4b9ee49b-0d5c-41e0-bb7c-be6332811880"       # os.getenv("SPACE_ID")


# ## Defining the Model object
# We need to define the Model object using the properties we defined so far:

model = Model(
    model_id = model_id,
    params = parameters,
    credentials = get_credentials(),
    project_id = project_id,
    space_id = space_id
    )


# ## Defining the inferencing input
# Foundation model inferencing API accepts a natural language input that it will use
# to provide the natural language response. The API is sensitive to formatting. Input
# structure, presence of training steps (one-shot, two-shot learning etc.), as well
# as phrasing all influence the final response and belongs to the emerging discipline of
# Prompt Engineering.
# 
# Let us provide the input we got from the Prompt Lab:

def main(cube_name, view_name):
    # Load TM1 data
    with TM1Service(address=tm1_credentials["service_root"],
                port=tm1_credentials["port"],
                user=tm1_credentials["username"],
                password=tm1_credentials["password"],
                ssl=False) as tm1:
        data = tm1.cubes.cells.execute_view(cube_name=cube_name, view_name=view_name,skip_zeros=False)
        df = tm1.cubes.cells.execute_view_dataframe_shaped(cube_name=cube_name, view_name=view_name,skip_zeros=False)
        df = df.fillna(0)

    headers = list(tm1.cubes.cells.execute_view_dataframe(cube_name=cube_name, view_name=view_name,skip_zeros=False))
    row_dimension, column_dimension = headers[0], headers[1]
    measure_dim = tm1.cubes.get_measure_dimension(cube_name=cube_name)
    measure_alias_names = tm1.elements.get_element_attribute_names(measure_dim,measure_dim)
    # row_alias_names = tm1.elements.get_element_attribute_names(row_dimension,row_dimension)
    # col_alias_names = tm1.elements.get_element_attribute_names(column_dimension,column_dimension)
    alias = tm1.elements.get_attribute_of_elements(measure_dim,measure_dim,measure_alias_names[1])

    if column_dimension == measure_dim:
        df.rename(columns=alias, inplace=True)                                                                                      # renommer les colonnes par leur alias
    elif row_dimension == measure_dim:
        df.replace({measure_dim: alias},inplace=True)
    df_md = df.to_markdown(index=False)
    question = tm1.cubes.cells.get_value(cube_name='TM1py_output',elements=[('TM1py_Scripts', 'GenAIAnalysis'), ('TM1py_outputs', 'Question')])
    role = tm1.cells.get_value(cube_name='TM1py_output',elements=[('TM1py_Scripts', 'GenAIAnalysis'), ('TM1py_outputs', 'Role')])
    
    list_context = tm1.cubes.cells.execute_view_ui_dygraph(cube_name=cube_name, view_name=view_name,skip_zeros=False)['titles'][0]['name']
    list_context = list_context.split(" / ")
    list_context = [alias[val] if (val in alias) else val for val in list_context]
    dimension_view_ordered_dict = tm1.cubes.cells.execute_view_raw(cube_name=cube_name, view_name=view_name,skip_zeros=False)['Cube']['Dimensions'][1:]
    dimension_view_ordered = [dico['Name'] for dico in dimension_view_ordered_dict]
    dimension_view_ordered.remove(row_dimension)
    dimension_view_ordered.remove(column_dimension)
    context = dict(zip(list_context[:-1],dimension_view_ordered))

    # df = build_pandas_dataframe_from_cellset(data, multiindex=False, sort_values=False)
    # # pivote le dataframe
    # non_measure_dim = tm1.cubes.get_dimension_names(cube_name=cube_name)
    # non_measure_dim.remove(measure_dim)
    # df = df.fillna(0)
    # # df['Values'] = df['Values'].apply(lambda x : format(float(x),'.2f'))
    # data_list = [{item: value} for item,value in data.items()]
    # df = df.rename({measure_dim:'KPI'},axis=1)
    # df_pivot = df.pivot(index=non_measure_dim, columns='KPI', values='Values').reset_index()
    # df_md = df_pivot.to_markdown(index=False)


    prompt_input = """Tu es un """ + role + """. A partir du contexte suivant: """ + str(context) + """ et du tableau suivant, répond à la question de manière claire et concise
    Entrée :
        | Indicator_Vector                                       |    2023.01 |     2023.02 |    2023.03 |     2023.04 |    2023.05 |     2023.06 |     2023.07 |     2023.08 |    2023.09 |    2023.10 |    2023.11 |     2023.12 |
    |:-------------------------------------------------------|-----------:|------------:|-----------:|------------:|-----------:|------------:|------------:|------------:|-----------:|-----------:|-----------:|------------:|
    | MARGE BRUTE                                            |  23548     |  14058      |  29026     |  18711      |  25831     |  32518      |  18403      |  21524      |  36303     |  25410     |  23927     |  42681      |
    | COUTS COMMERCIAUX                                      |   -532     |   -555      |   -533     |   -563      |   -529     |   -193      |   -602      |   -409      |   -408     |   -428     |   -442     |   -116      |
    | COUTS GENERAUX ET ADMINISTRATIFS                       | -18477     | -20028      | -18669     | -18319      | -18058     | -18446      | -17642      | -17445      | -17417     | -17813     | -18239     | -19039      |
    | AUTRES CHARGES ET PRODUITS OPERATIONNELS               |  -5614     |  -5578      |  -5594     |  -5578      |  -5566     |  -5637      |  -5556      |  -5566      |  -5628     |  -5591     |  -5546     |  -5603      |
    | RESULTAT DES CO-ENTREPRISES (à compter de 2013)        |    141.832 |     53.3437 |    170.308 |     53.3437 |    135.333 |    180.307  |    -30.1452 |    -72.1396 |    184.806 |    518.837 |    138.832 |    -13.1673 |
    | RESULTAT DES ENTREPRISES ASSOCIEES (à compter de 2013) |     13     |     16      |     -6.5   |     16      |     10.5   |    -33.9932 |     10.5    |     12      |      5.5   |     10.5   |     13     |      9      |

    Quelle est le marge brute moyenne du troisième trimestre ?
    Sortie :
    25410
    =(18403+21524+36303)/3
    Entrée :
        | Indicator_Vector                                                 |   2023.01 |   2023.02 |   2023.03 |   2023.04 |   2023.05 |   2023.06 |   2023.07 |   2023.08 |   2023.09 |   2023.10 |   2023.11 |   2023.12 |
    |:-----------------------------------------------------------------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
    | CESSIONS INDUSTRIELLES                                           |       202 |        38 |      2349 |        82 |       119 |      1552 |       181 |       102 |      3590 |       470 |      9377 |       594 |
    | REMBOURSEMENT D'ACTIFS FINANCIERS I12 ET I4                      |      4262 |       790 |       820 |       814 |       823 |      7259 |      1290 |       624 |       427 |       416 |     22713 |      2978 |
    | CDV - Plus ou moins values de cessions s/ Immo. Corp. & Incorp.  |     -1329 |        39 |        63 |        83 |        78 |       408 |     -1023 |       594 |       363 |       142 |       239 |       423 |
    | G&A - Plus ou moins values de cessions s/ Immo. Corp. &  Incorp. |         0 |         0 |         0 |         0 |        -3 |         0 |       521 |      -341 |         0 |         0 |         0 |         0 |
    | INVESTISSEMENTS - Maintenance & Croissance                       |    -22457 |    -18055 |    -16169 |    -13365 |     -6984 |    -19641 |    -13204 |    -23327 |    -38944 |    -23743 |    -23815 |    -44880 |
    | CAPACITE D'AUTOFINANCEMENT (par niveaux)                         |     12857 |      2083 |     18735 |      8738 |     16408 |     22286 |      9285 |     12836 |     25055 |     15972 |     13843 |     25714 |

    Quel sont les 3 meilleurs mois en terme d'autofinancement ?
    Sortie :
    Décembre
    Septembre
    Juin
    Entrée :
    """ + str(df_md) + """

    """ + str(question) + """
    
    Sortie :"""


    # Entrée :
    # | Subsidiaries       | Channels     | Products   | Months     | Versions         |   Quantity |   Unit Cost |   Unit Sale Price |
    # |:-------------------|:-------------|:-----------|:-----------|:-----------------|-----------:|------------:|------------------:|
    # | GO AMERICAS REGION | ALL CHANNELS | Lanterns   | Apr        | Budget Version 1 |      23040 |     20.916  |           28.8114 |
    # | GO AMERICAS REGION | ALL CHANNELS | Lanterns   | Aug        | Budget Version 1 |      14309 |     21.1884 |           29.1867 |
    # | GO AMERICAS REGION | ALL CHANNELS | Lanterns   | Dec        | Budget Version 1 |      54416 |     21.1561 |           29.1421 |
    # | GO AMERICAS REGION | ALL CHANNELS | Lanterns   | Feb        | Budget Version 1 |      19757 |     21.4532 |           29.5514 |
    # | GO AMERICAS REGION | ALL CHANNELS | Lanterns   | Jan        | Budget Version 1 |      17001 |     21.3269 |           29.3775 |
    # | GO AMERICAS REGION | ALL CHANNELS | Lanterns   | Jul        | Budget Version 1 |      33410 |     21.6072 |           29.7636 |
    # | GO AMERICAS REGION | ALL CHANNELS | Lanterns   | Jun        | Budget Version 1 |      10700 |     21.9791 |           30.2759 |
    # | GO AMERICAS REGION | ALL CHANNELS | Lanterns   | Mar        | Budget Version 1 |      92158 |     21.7897 |           30.015  |
    # | GO AMERICAS REGION | ALL CHANNELS | Lanterns   | May        | Budget Version 1 |      14835 |     21.5703 |           29.7127 |
    # | GO AMERICAS REGION | ALL CHANNELS | Lanterns   | Nov        | Budget Version 1 |      38138 |     21.1822 |           29.1782 |
    # | GO AMERICAS REGION | ALL CHANNELS | Lanterns   | Oct        | Budget Version 1 |      35839 |     21.2263 |           29.2389 |
    # | GO AMERICAS REGION | ALL CHANNELS | Lanterns   | Sep        | Budget Version 1 |       8600 |     21.2982 |           29.3379 |
    # | GO AMERICAS REGION | ALL CHANNELS | Lanterns   | Total Year | Budget Version 1 |     362203 |     21.4235 |           29.5105 |

    # Quelle a été le bénéfice total du troisième trimestre ?
    # Sortie :
    # 456093.309
    # (=33410 *(29.7636 - 21.6072)
    # + 14309 *(29.1867 - 21.1884)
    # + 8600 *(29.3379 - 21.2982))

    # Entrée :
    # """ + str(df_md) + """

    # """ + str(question) + """

    # Sortie :
    # """

    # ## Execution
    # Let us now use the defined Model object and pair it with input and
    # generate the response:
    print(prompt_input)
    print("Submitting generation request...")
    try:
        generated_response = model.generate_text(prompt=prompt_input, guardrails=True)
    except Exception as error:
        input_string = error.error_msg
        match = re.search(r'"message":"(.*?)","more_info"', input_string)

        # Vérifier si un match est trouvé et extraire la sous-chaîne
        if match:
            extracted_substring = match.group(1)
            if ("tokens" in extracted_substring) and ("exceed" in extracted_substring):
                generated_response = 'Le nombre de jetons en entrée dépasse la limite totale de jetons pour ce modèle.'
        else:
            generated_response = "Une erreur est survenue."

    print(generated_response)

    # write value in response cell

    with TM1Service(address=tm1_credentials["service_root"],
                port=tm1_credentials["port"],
                user=tm1_credentials["username"],
                password=tm1_credentials["password"],
                ssl=False) as tm1:
        cube_name = 'TM1py_output'
        cube_dimensions_names = tm1.cubes.get_dimension_names(cube_name=cube_name)

        tm1.cells.write_value(
            generated_response,
            cube_name=cube_name,
            element_tuple=['GenAIAnalysis', 'Answer']
            )

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <arg1> <arg2>")
    else:
        main(sys.argv[1], sys.argv[2])
    # main('RP_Vector','zViewSource_Country')
