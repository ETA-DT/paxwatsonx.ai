import sys

#!/usr/bin/env python
# coding: utf-8

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
        wml_url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2024-05-10"
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
        "apikey" : "dHWFYaT3oUsopTavFA21SBnUXsMaDUxVUWsMAZfwILs0" # getpass.getpass("Please enter your api key (hit enter): ")
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
    "min_new_tokens": 20,
    "max_new_tokens": 800,
    "repetition_penalty": 1
}


# ## Defining the project id or space id
# The API requires project id or space id that provides the context for the call. We will obtain
# the id from the project or space in which this notebook runs:


project_id = "6c65d7af-9df1-4cc0-bcab-857f4d936ac8"     # os.getenv("PROJECT_ID")
space_id = "9a541e05-66a4-462e-a7f4-892acae9ea06"       # os.getenv("SPACE_ID")

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

    # Load TM1 data arranged
    with TM1Service(address=tm1_credentials["service_root"],
                port=tm1_credentials["port"],
                user=tm1_credentials["username"],
                password=tm1_credentials["password"],
                ssl=False) as tm1:
        data = tm1.cubes.cells.execute_view(cube_name=cube_name, view_name=view_name,skip_zeros=False)
        df = build_pandas_dataframe_from_cellset(data, multiindex=False, sort_values=False)

    non_measure_dim = tm1.cubes.get_dimension_names(cube_name=cube_name)
    measure_dim = tm1.cubes.get_measure_dimension(cube_name=cube_name)
    non_measure_dim.remove(measure_dim)
    data_list = [{item: value} for item,value in data.items()]
    df['Values'] = df['Values'].apply(lambda x : format(float(x),'.2f'))
    df = df.rename({measure_dim:'KPI'},axis=1)
    df_pivot = df.pivot(index=non_measure_dim, columns='KPI', values='Values').reset_index()
    df_md = df_pivot.to_markdown(index=False)

    prompt_input = """Tu es un analyste financier. Donne moi uniquement l'analyse des points clés de ce tableau de manière claire et concise, et appuie des propos à l'aide des données

    """ + str(df_md)
    # ## Execution
    # Let us now use the defined Model object and pair it with input and
    # generate the response:
    print("Submitting generation request...")
    generated_response = model.generate_text(prompt=prompt_input, guardrails=True)
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
            element_tuple=['GenAIAnalysis', 'Results']
            )

    file = open('myfile.txt', 'w')
    file.write(generated_response)
    file.close()
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <arg1> <arg2>")
    else:
        main(sys.argv[1], sys.argv[2])




