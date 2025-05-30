import requests
import json

import requests
import pandas as pd

base_url = "https://services.odata.org/TripPinRESTierService/"
endpoint = "People"
'''
$format: returns JSON results.
$top: limits how many results you get.
$filter: asks the service for only those people who have at least one trip (any(t: ...)) with a budget greater than 3000.
'''
params = {
    '$format': 'json',
    '$top': 2,
    '$filter': "Trips/any(t: t/Budget gt 3000)"
}

response = requests.get(base_url + endpoint, params=params)
data = response.json()

print(data)

with open('./json_files/odata_response.json', 'w') as outfile:
    json.dump(data, outfile)


# importing json file to dataframe
# with open('./json_files/odata_response.json', 'r') as openfile:
#     json_data = json.load(openfile)

df = pd.read_json('./json_files/odata_response.json')
print(df)
