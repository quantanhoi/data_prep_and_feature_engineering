import requests
import json

import requests

base_url = "https://services.odata.org/TripPinRESTierService/"
endpoint = "People"
params = {
    '$format': 'json',
    '$top': 2,
    '$filter': "Trips/any(t: t/Budget gt 3000)"
}

response = requests.get(base_url + endpoint, params=params)
data = response.json()

print(data)

with open('odata_response.json', 'w') as outfile:
    json.dump(data, outfile)


    