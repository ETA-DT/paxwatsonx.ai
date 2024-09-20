#!/usr/bin/env python
# coding: utf-8

# import libraries
import ast
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
        wml_url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2024-05-29"
        Headers = {
            "Authorization": "Bearer " + self.access_token, 
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        data = {
            "model_id" : model_id,
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
tm1_user = "erwan.tang"                     # getpass.getpass('Please enter your TM1 username: ')
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
        "apikey" : "cvtmyz9Fof7pwnIAgc9vQ1Gar9ctz-MBlsLcZg1GKgy_" # getpass.getpass("Please enter your api key (hit enter): ") # replace with new watsonx.ai API (from IAM)
    }


## Inferencing

# Defining the model id
model_id = "meta-llama/llama-3-70b-instruct"
extract_model_id = "mistralai/mixtral-8x7b-instruct-v01"

# Defining the model parameters

parameters = {
    "decoding_method": "greedy",
    "min_new_tokens": 20,
    "max_new_tokens": 800,
    "repetition_penalty": 1.1,
    "stop_sequences": ["'\'''\'''\''","\"\"\"","```"]
}

extract_model_parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 50,
    "min_new_tokens": 1,
    "stop_sequences": ["\n\n"],
    "repetition_penalty": 1
}

## Defining the project id or space id
# The API requires project id or space id that provides the context for the call. We will obtain
# the id from the project or space in which this notebook runs:

project_id = "676c7e36-db45-4ebd-bb23-6a7824b815b4"       # os.getenv("PROJECT_ID") # replace with new watsonx.ai project_id
# space_id = "4b9ee49b-0d5c-41e0-bb7c-be6332811880"       # os.getenv("SPACE_ID")

## Defining the Model object

model = Model(
    model_id = model_id,
    params = parameters,
    credentials = get_credentials(),
    project_id = project_id
    )

extract_model = Model(
	model_id = extract_model_id,
	params = extract_model_parameters,
	credentials = get_credentials(),
	project_id = project_id
    )

# main function to execute from process
def main(cube_name, view_name, selected_prompt):

    # # Load TM1 data arranged
    # with TM1Service(address=tm1_credentials["service_root"],
    #             port=tm1_credentials["port"],
    #             user=tm1_credentials["username"],
    #             password=tm1_credentials["password"],
    #             ssl=False) as tm1:
    #     data = tm1.cubes.cells.execute_view(cube_name=cube_name, view_name=view_name,skip_zeros=False)
    #     df = build_pandas_dataframe_from_cellset(data, multiindex=False, sort_values=False)
    
    # Vue directe                                                                                       # renommer les colonnes par leur alias

    with TM1Service(address=tm1_credentials["service_root"],                                           
                port=tm1_credentials["port"],
                user=tm1_credentials["username"],
                password=tm1_credentials["password"],
                ssl=False) as tm1:                                                                      # définir et se connecter à l'instance tm1
        data = tm1.cubes.cells.execute_view(cube_name=cube_name, view_name=view_name,skip_zeros=False)  # vue de cube
        df = tm1.cubes.cells.execute_view_dataframe_shaped(cube_name=cube_name, view_name=view_name,skip_zeros=False)
        df = df.fillna(0)

    headers = list(tm1.cubes.cells.execute_view_dataframe(cube_name=cube_name, view_name=view_name,skip_zeros=False))
    row_dimension, column_dimension = headers[0], headers[1]
    measure_dim = tm1.cubes.get_measure_dimension(cube_name=cube_name)
    country_dim = 'Pays'

    # vérification d'existence d'alias pour la dimension indicateur (autre que l'attribut format)
    measure_alias_names = tm1.elements.get_element_attribute_names(measure_dim,measure_dim)
    alias_bool = False
    if (len(measure_alias_names) > 0) and (len(measure_alias_names) != 1 or (measure_alias_names[0].lower() != "format")):
        alias_bool = True
        alias = tm1.elements.get_attribute_of_elements(measure_dim,measure_dim,measure_alias_names[1])                                  # choix de l'alias à envoyer au LLM
        if column_dimension == measure_dim:
            df.rename(columns=alias, inplace=True)                                                                                      # renommer les colonnes par leur alias
        elif row_dimension == measure_dim:
            df.replace({measure_dim: alias},inplace=True)

    df_md = df.to_markdown(index=False)                                                                                                 # formatage du dataframe en string
    role = tm1.cells.get_value(cube_name='TM1py_output',elements=[('TM1py_Scripts', 'GenAIAnalysis'), ('TM1py_outputs', 'Role')])       # role selectionné dans la picklist tm1

    def get_context(cube_name,view_name):
        list_context = tm1.cubes.cells.execute_view_ui_dygraph(cube_name=cube_name, view_name=view_name,skip_zeros=False)['titles'][0]['name']
        list_context = list_context.split(" / ")
        return list_context
    
    list_context = get_context(cube_name,view_name)
    
    if alias_bool:
        list_context = [alias[val] if (val in alias) else val for val in list_context]

    # récupérer les noms de dimension du contexte dans l'ordre de la vue
    mdx_query = tm1.cubes.views.get_mdx_view(cube_name,view_name).MDX
    # Regex pour capturer les dimensions après WHERE
    where_pattern = r'WHERE\s*\((.*?)\)'
    # Extraire tout ce qui est entre les parenthèses après WHERE
    where_clause = re.search(where_pattern, mdx_query, re.DOTALL).group(1)
    # Trouver toutes les dimensions dans la clause WHERE
    dimensions = re.findall(r'\[([^\]]+)\]\.\[([^\]]+)\]', where_clause)
    # Extraire les noms des dimensions (le premier élément de chaque tuple) dans l'ordre de la vue
    dimension_view_names = [dim[0] for dim in dimensions]
    measure_dim_level_max = tm1.elements.get_levels_count(measure_dim,measure_dim)
    other_indicators = tm1.elements.get_elements_by_level(measure_dim,measure_dim,level=0)      # liste des autres indicateurs de la dimension indicateurs
    countries = tm1.elements.get_elements_by_level(country_dim,country_dim,level=0)             # liste des pays de la dimension Pays
    if '% Marketing' in other_indicators: other_indicators.remove('% Marketing')

    def extract_indicators_from_text(text):
        # Regex pour capturer les noms d'indicateurs après les puces
        pattern = r">\s*(.*)"
        found_indicators = re.findall(pattern, text)
        return found_indicators

    def match_indicators(found_indicators, reference_indicators):
        # Comparer les indicateurs extraits avec ceux de la liste de référence
        matched_indicators = [indicator for indicator in found_indicators if indicator in reference_indicators]
        return matched_indicators

    # Extraire les indicateurs cités dans le texte
    # Identifier les indicateurs qui sont aussi dans la liste de référence
    previous_indicators = tm1.cells.get_value(cube_name='TM1py_output', elements=[('TM1py_Scripts', 'GenAIAnalysis'), ('TM1py_outputs', 'PreviousIndicators')])
    previous_results = tm1.cells.get_value(cube_name='TM1py_output', elements=[('TM1py_Scripts', 'GenAIAnalysis'), ('TM1py_outputs', 'PreviousResults')])
    
    print('\nprevious_results\n')
    print(previous_results)

    if previous_indicators:
        previous_indicators_list = ast.literal_eval(previous_indicators)
    else:
        previous_indicators_list = []
    other_indicators = [indicator for indicator in other_indicators if indicator not in previous_indicators_list]

    context = dict(zip(dimension_view_names,list_context))          # Eléments des dimensions de contexte
    
    input_1 = """
            donne-moi une analyse détaillée des performances selon les indicateurs financiers pour chaque mois.\
            N'inclus pas de préfixe dans la réponse comme 'Voici ma réponse' ou 'Voici mon analyse'
            """
    input_2 = """\
            fournis seulement un résumé des performances des indicateurs financiers pour l'année.\
            N'inclus pas de préfixe dans la réponse comme 'Voici ma réponse', 'Voici un résumé'"
            """
    input_3 = """\
            donne-moi seulement les principaux enseignements à tirer des indicateurs financiers suivants.\
            N'inclus pas de préfixe dans la réponse comme 'Voici ma réponse', 'Voici les informations'"
            """
    input_4 = " donne-moi les informations clées du contenu du tableau et les recommandations d'action les plus pertinentes pour augmenter les performances de mon entreprise."\
            f"Les recommandations d'action doivent être explicites et chiffrées afin de pouvoir appliquer une solution à impact direct en exploitant seulement les indicateurs {other_indicators} qui ne sont pas inclus dans ce tableau."\
            "N'inclus pas de préfixe dans la réponse comme 'Voici ma réponse', 'Voici les informations clées'"\
            f"Résume moi ensuite ta réponse sous forme de liste à puces pour les 3 indicateurs les plus impactant. Conserve la syntaxe exacte du noms des indicateurs {other_indicators}"
    input_5 = """\
            donne-moi seulement une comparaison entre les prévisions et l'actuel, et interprète moi les écarts avec des explications chiffrées.\
            N'inclus pas de préfixe dans la réponse comme 'Voici ma réponse', 'Voici la comparaison'" 
            """
    input_6 = f"Donne-moi une analyse des informations clées des données du tableau et identifie les top 2 recommandations d'action les plus pertinentes pour augmenter les performances de mon entreprise. Ne cible pas les mêmes indicateurs que précédemment. "\
            f"Indicateurs ciblés précedemment : {previous_indicators}. "\
            f"Les recommandations doivent être explicites, chiffrées, et directement applicables (par exemple, augmenter campagne marketing de x % du pays P) en utilisant uniquement les indicateurs spécifiques de la liste {other_indicators} qui ne figurent pas dans le tableau. "\
            "Conserve exactement la syntaxe des indicateurs. N'inclus pas de préfixe dans ta réponse (par exemple, 'Voici ma réponse'). "\
            # f"Après l'analyse, résume les recommandations citées sous forme de liste à puces, en utilisant exactement la syntaxe des noms des indicateurs présents dans {other_indicators}, et assure-toi que chaque puce corresponde à un indicateur précis. "

    inputs = {"Analyse mensuelle" : input_1, "Resume des performances" : input_2, "Principaux enseignements" : input_3, "Informations clees et recommandations" : input_6, "Comparaison actuel et forecast" : input_5}

    prompt_input = """Tu es un """ + role + """. A partir du contexte général suivant: """ + str(context) + """ et du tableau suivant, """ + inputs[selected_prompt] + """
    """ + str(df_md)

    # ## Execution
    # Let us now use the defined Model object and pair it with input and
    # generate the response:
    # print(prompt_input)

    print("Submitting generation request...")
    generated_response = model.generate_text(prompt=prompt_input, guardrails=False)
    print('\nGenerated_response\n')
    print(generated_response)    

    extract_prompt_input = f"""Voici une liste d'indicateurs:
                            [Campagne Marketing, Programme Fidélité, Couts Commerciaux, Couts des ventes, Couts généraux, Ventes Carburant, Ventes de pièces détachées, Prestation atelier, Recettes commerciales, Recette passager, Couts Stock Biens, Frais de personnel (CD), Coûts des accidents du travail, Réparations, Indemnisation tiers, Frais de voyages (SO), Coût Bio carburant, Coût Gaz, Pneumatiques, Recettes commerciales- engagement contractuel, Revenu des activités de tourisme (occasionnel)]
                            Extrais de manière exhaustive tous les indicateurs qui apparaissent dans ce texte. Si tu ne trouves pas d'indicateurs, ne retourne rien. Conserve exactement le nom des indicateurs de la liste.

                            Texte: L'analyse des informations clées du contenu du tableau montre que les pays ont des tendances différentes en ce qui concerne leurs ventes. Certains pays comme la Finlande et l'Irlande ont des ventes élevées tandis que d'autres comme le Royaume-Uni et les Pays-Bas ont des ventes plus faibles.Les deux recommandations d'action les plus pertinentes pour augmenter les performances de votre entreprise sont :• Augmenter Campagne Marketing de 20% pour améliorer la visibilité de vos produits et services sur les marchés internationaux, en particulier en Finlande et en Irlande où les ventes sont élevées.• Réduire Couts des ventes de 15% en optimisant les processus logistiques et en renégociant les contrats avec les fournisseurs, notamment en Espagne et au Portugal où les coûts des ventes sont élevés.
                            Indicateurs trouvés: > Campagne Marketing
                            > Couts des ventes

                            Texte: Analyse des informations clées du contenu du tableau :Le tableau présente les données de ventes mensuelles pour différents pays européens. Les valeurs sont exprimées en unités monétaires (probablement euros). Les données montrent une grande variabilité entre les pays et les mois.Les pays avec les ventes les plus élevées sont l'Irlande, l'Allemagne et la Finlande. Les pays avec les ventes les plus faibles sont la Belgique et le Royaume-Uni.Il est important de noter que certaines valeurs sont négatives, ce qui peut indiquer des pertes ou des coûts associés aux ventes.Recommandations d'action :* Augmenter le Programme Fidélité de 20% pour améliorer la rétention des clients et encourager les achats répétés.* Réduire les Couts Commerciaux de 15% pour améliorer la marge bénéficiaire et compétitivité. 
                            Indicateurs trouvés: > Programme Fidélité
                            > Couts Commerciaux

                            Texte: {generated_response}
                            Indicateurs trouvés: """

    extracted_indicators = extract_model.generate_text(prompt=extract_prompt_input, guardrails=False)
    found_indicators = extract_indicators_from_text(extracted_indicators)
    matched_indicators = match_indicators(found_indicators, other_indicators)
    matched_indicators_picklist = "static::" + ":".join(matched_indicators)

    print('\nextracted_indicators\n')
    print(extracted_indicators)
    print('\nfound_indicators\n')
    print(found_indicators)
    print('\nmatched_indicators\n')
    print(matched_indicators)

    extract_prompt_input_countries = f"""Extrais de manière exhaustive tous les pays qui apparaissent dans ce texte.


                    Texte: L'analyse des informations clées du contenu du tableau montre que les pays ont des tendances différentes en ce qui concerne leurs ventes. Certains pays comme la Finlande et l'Irlande ont des ventes élevées tandis que d'autres comme le Royaume-Uni et les Pays-Bas ont des ventes plus faibles.Les deux recommandations d'action les plus pertinentes pour augmenter les performances de votre entreprise sont :• Augmenter Campagne Marketing de 20% pour améliorer la visibilité de vos produits et services sur les marchés internationaux, en particulier en Finlande et en Irlande où les ventes sont élevées.• Réduire Couts des ventes de 15% en optimisant les processus logistiques et en renégociant les contrats avec les fournisseurs, notamment en Espagne et au Portugal où les coûts des ventes sont élevés.
                    Pays trouvés: > Finlande
                    > Irlande
                    > Pays-Bas
                    > Royaume-Uni

                    Texte: Analyse des informations clées du contenu du tableau :Le tableau présente les données de ventes mensuelles pour différents pays européens. Les valeurs sont exprimées en unités monétaires (probablement euros). Les données montrent une grande variabilité entre les pays et les mois.Les pays avec les ventes les plus élevées sont l'Irlande, l'Allemagne et la Finlande. Les pays avec les ventes les plus faibles sont la Belgique et le Royaume-Uni.Il est important de noter que certaines valeurs sont négatives, ce qui peut indiquer des pertes ou des coûts associés aux ventes.Recommandations d'action :* Augmenter le Programme Fidélité de 20% pour améliorer la rétention des clients et encourager les achats répétés.* Réduire les Couts Commerciaux de 15% pour améliorer la marge bénéficiaire et compétitivité. 
                    Pays trouvés: > Allemagne
                    > Belgique
                    > Finlande
                    > Pays-Bas
                    > Royaume-Uni

                    Texte: {generated_response}
                    Pays trouvés:"""
    
    extracted_countries = extract_model.generate_text(prompt=extract_prompt_input_countries, guardrails=False)
    found_countries = extract_indicators_from_text(extracted_countries)
    matched_countries = match_indicators(found_countries, countries)
    matched_countries_picklist = "static::" + ":".join(matched_countries)

    print('\nextracted_countries\n')
    print(extracted_countries)
    print('\nfound_countries\n')
    print(found_countries)
    print('\nmatched_countries\n')
    print(matched_countries)

    extract_prompt_input_percent = f"""Voici une liste d'indicateurs:
                            [Campagne Marketing, Programme Fidélité, Couts Commerciaux, Couts des ventes, Couts généraux, Ventes Carburant, Ventes de pièces détachées, Prestation atelier, Recettes commerciales, Recette passager, Couts Stock Biens, Frais de personnel (CD), Coûts des accidents du travail, Réparations, Indemnisation tiers, Frais de voyages (SO), Coût Bio carburant, Coût Gaz, Pneumatiques, Recettes commerciales- engagement contractuel, Revenu des activités de tourisme (occasionnel)]
                            Extrais de manière exhaustive tous les indicateurs et le pourcentage et pays associé qui sont recommandés de modifier dans ce texte. Si tu ne trouves pas d'indicateurs, ne retourne rien. Conserve exactement le nom des indicateurs de la liste. Si tu ne trouves pas de pays associé à cet indicateur, renvoie "tout".

                            Texte: L'analyse des informations clées du contenu du tableau montre que les pays ont des tendances différentes en ce qui concerne leurs ventes. Certains pays comme la Finlande et l'Irlande ont des ventes élevées tandis que d'autres comme le Royaume-Uni et les Pays-Bas ont des ventes plus faibles.Les deux recommandations d'action les plus pertinentes pour augmenter les performances de votre entreprise sont :• Augmenter Campagne Marketing de 20% pour améliorer la visibilité de vos produits et services sur les marchés internationaux, en particulier en Finlande et en Irlande où les ventes sont élevées.• Réduire Couts des ventes de 15% en optimisant les processus logistiques et en renégociant les contrats avec les fournisseurs, notamment en Espagne et au Portugal où les coûts des ventes sont élevés.
                            Indicateurs trouvés: > Campagne Marketing;;20;;tout
                            > Couts des ventes;;15;;tout

                            Texte: Analyse des informations clées du contenu du tableau :Le tableau présente les données de ventes mensuelles pour différents pays européens. Les valeurs sont exprimées en unités monétaires (probablement euros). Les données montrent une grande variabilité entre les pays et les mois.Les pays avec les ventes les plus élevées sont l'Irlande, l'Allemagne et la Finlande. Les pays avec les ventes les plus faibles sont la Belgique et le Royaume-Uni.Il est important de noter que certaines valeurs sont négatives, ce qui peut indiquer des pertes ou des coûts associés aux ventes.Recommandations d'action :* Augmenter le Programme Fidélité de 38% de l'Irlande pour améliorer la rétention des clients et encourager les achats répétés.* Réduire les Couts Commerciaux de 17% pour améliorer les Recettes Commerciales, la marge bénéficiaire et la compétitivité. 
                            Indicateurs trouvés: > Programme Fidélité;;38;;Irlande
                            > Couts Commerciaux;;17;;tout

                            Texte: {generated_response}
                            Indicateurs trouvés: """
    
    extracted_percent = extract_model.generate_text(prompt=extract_prompt_input_percent, guardrails=False)
    found_percent = extract_indicators_from_text(extracted_percent)
    # matched_percent_picklist = "static::" + ":".join(found_percent)

    print('\nextracted_percent\n')
    print(extracted_percent)
    print('\nfound_percent\n')
    print(found_percent)

    indicator_percent_country = []
    for association in found_percent:
        indicator,percent,pays = association.split(";;")
        indicator_percent_country.append({'indicator':indicator,'percent':percent,'country':pays})

    print(indicator_percent_country)


    def update_subset(subset_name,dimension_name,hierarchy_name,matched):
        indicator_subset = tm1.subsets.get(subset_name,dimension_name,hierarchy_name)
        indicator_subset.elements = []
        tm1.subsets.update(indicator_subset)
        indicator_subset.add_elements(matched)
        tm1.subsets.update(indicator_subset)

    
    print('SUBSET Indicateur AVANT UPDATE')
    print(tm1.subsets.get_element_names('Indicateurs_Activité','Indicateurs_Activité','IndicatorToModify'))
    update_subset('IndicatorToModify','Indicateurs_Activité','Indicateurs_Activité',matched_indicators)
    print('SUBSET Indicateur APRES UPDATE')
    print(tm1.subsets.get_element_names('Indicateurs_Activité','Indicateurs_Activité','IndicatorToModify'))
    

    print('SUBSET PAYS TROUVES AVANT UPDATE')
    print(tm1.subsets.get_element_names('Pays','Pays','PaysExtraits'))
    update_subset('PaysExtraits','Pays','Pays',[cell['country'] for cell in indicator_percent_country])

    print('SUBSET PAYS TROUVES APRES UPDATE')
    print(tm1.subsets.get_element_names('Pays','Pays','PaysExtraits'))


    # write value in response cell

    with TM1Service(address=tm1_credentials["service_root"],
                port=tm1_credentials["port"],
                user=tm1_credentials["username"],
                password=tm1_credentials["password"],
                ssl=False) as tm1:
        output_cube_name = 'TM1py_output'
        cube_dimensions_names = tm1.cubes.get_dimension_names(cube_name=cube_name)

        tm1.cells.write_value(
            generated_response,
            cube_name=output_cube_name,
            element_tuple=['GenAIAnalysis', 'Results']
            )
        
        tm1.cells.write_value(
            matched_indicators,
            cube_name=output_cube_name,
            element_tuple=['GenAIAnalysis', 'CurrentIndicators']
            )
        
        tm1.cells.write_value(
            matched_indicators_picklist,
            cube_name="}PickList_"+output_cube_name,
            element_tuple=['GenAIAnalysis', 'CurrentIndicatorsPickList', 'Value']
            )
        
        
        for target in indicator_percent_country:
            percent, target_country, target_indicator = target['percent'],target['country'],target['indicator']
            print(percent, target_country, target_indicator)
            for period in tm1.subsets.get_element_names('Period','Period','2024_mois'):                                 # à généraliser pour récupérer le subset de period d'une vue donnée
                tm1.cells.write_value(
                    int(percent),
                    cube_name=cube_name,
                    element_tuple=['BUDG_VC_AJUST', period, target_country, target_indicator]
                    )
            
            
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: script.py <arg1> <arg2> <arg3>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])

