import requests
import json


#converting response to json
response = requests.get('https://petstore.swagger.io/v2/pet/findByStatus?status=sold').json()
# print(response)

# Writing JSON to a file in Python using json.dump() 
with open('./json_files/petstore_response.json', 'w') as outfile:    #'w' for writing
    json.dump(response, outfile)
    
    
# Reading JSON from a file using json.load()
with open('./json_files/petstore_response.json', 'r') as openfile:   #'r' for reading
    json_object = json.load(openfile)
    
print(json_object)


