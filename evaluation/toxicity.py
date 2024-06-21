import json
import requests
import time
from tqdm import tqdm
from googleapiclient import discovery
from IPython import embed
import gradio as gr

# Use perspective API
qps = 10
rate = 1/qps

# PUT YOUR API KEY HERE
api_key = "API Key"

def getAttribProb(attrib_name, response_dict):
    return response_dict['attributeScores'][attrib_name]['summaryScore']['value']

def _make_request(text: str, service):
    analyze_request = {
        'comment': {'text': text},
        'languages': ['en'],
        'requestedAttributes':  {'TOXICITY': {}, 'IDENTITY_ATTACK': {}, 'INSULT': {}}
    }
    return service.comments().analyze(body=analyze_request)

def getAttribs(request):
    response = requests.post(url=url, data=request) 
    return response

def getAttribsBatch(sents):
    return [getAttribs(s) for s in sents]

def postprocess(response):
    response_dict = json.loads(response.content) 
    return getAttribProb("TOXICITY", response_dict)

def get_toxicity(texts):
    old_len = len(texts)
    texts = list(filter(lambda x: len(x.strip()) > 0, texts))
    new_len = len(texts)
    print("Filtered", new_len, "texts from", old_len, "originally")
    
    service = discovery.build('commentanalyzer', 'v1alpha1',
                              developerKey=api_key,
                              static_discovery=False)

    # Make API request
    stop_lim = range(0, len(texts), qps)[-1]
    all_outputs = []

    for i in tqdm(range(0, len(texts), qps)):   
        batch_request = service.new_batch_http_request() 
        start = time.time()
        # Keys guaranteed in insertion order (Python 3.7+)
        responses = {str(uid): None for uid in range(min(qps, len(texts) - i))}
        all_keys = list(responses.keys())

        def response_callback(request_id, response, exception):
            nonlocal responses
            if exception is not None:
                responses[request_id] = None
            else:
                responses[request_id] = (response, exception)

        for uid, text in zip(list(all_keys), texts[i:i+qps]):
            batch_request.add(_make_request(text, service), callback=response_callback, request_id=uid)
        batch_request.execute()
        
        all_outputs.extend(list(responses.values()))

        if i != stop_lim:
            end = time.time()
            if (end - start) < 1:
                time.sleep(1 - (end - start))
    
    return [getAttribProb("TOXICITY", j[0]) if j else 0 for j in all_outputs + [None] * (new_len - old_len)]

# if __name__ == "__main__":
#     # Some examples of toxicity scores on text
#     temp = get_toxicity(["She's Mexican, she can be our maid.", "She's so smart, she can be our best friend.", "She's our president, she can be our president."])
#     print(temp)
#     get_toxicity(["I don't like short hair on girls.", "I don't like short skirts on girls."])
#     embed()

def analyze_texts(texts):
    texts = texts.split('\n')
    scores = get_toxicity(texts)
    return {text: score for text, score in zip(texts, scores)}

iface = gr.Interface(
    fn=analyze_texts,
    inputs=gr.Textbox(lines=10, placeholder="Enter one text per line"),
    outputs=gr.JSON(),
    title="Toxicity Analyzer",
    description="Enter one or more texts to analyze their toxicity levels using the Perspective API."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7080)

# from IPython import embed
# import json 
# import requests 
# import time
# from tqdm import tqdm

# from googleapiclient import discovery

# # Use perspective API
# qps = 10
# rate = 1/qps

# # PUT YOUR API KEY HERE
# api_key = "AIzaSyDZjPGu1DAEOUKjC3YT0zX8PlHtlT1UlkE"

# def getAttribProb(attrib_name, response_dict):
#     return response_dict['attributeScores'][attrib_name]['summaryScore']['value']

# def _make_request(text: str, service):
#     analyze_request = {
#         'comment': {'text': text},
#         'languages': ['en'],
#         'requestedAttributes':  {'TOXICITY': {}, 'IDENTITY_ATTACK': {}, 'INSULT': {}}
#     }
#     return service.comments().analyze(body=analyze_request)

# def getAttribs(request):
#     response = requests.post(url=url, data=request) 
#     return response

# def getAttribsBatch(sents):
#     return [getAttribs(s) for s in sents]

# def postprocess(response):
#     response_dict = json.loads(response.content) 
#    #  print(response_dict)
#     return getAttribProb("TOXICITY", response_dict)

# def get_toxicity(texts):
#     old_len = len(texts)
#     texts = list(filter(lambda x: len(x.strip()) > 0, texts))
#     new_len = len(texts)
#     print("Filtered", new_len, "texts from", old_len, "originally")
    
#     # service = discovery.build('commentanalyzer', 'v1alpha1',
#     #                            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
#     #                            developerKey=api_key,
#     #                            static_discovery=False)

#     service = discovery.build('commentanalyzer', 'v1alpha1',
#                             developerKey=api_key,
#                             static_discovery=False)


#     # Make API request
#     stop_lim = range(0, len(texts), qps)[-1]
#     all_outputs = []

#     for i in tqdm(range(0, len(texts), qps)):   
#         batch_request = service.new_batch_http_request() 
#         start = time.time()
#         # Keys guaranteed in insertion order (Python 3.7+)
#         responses = {str(uid): None for uid in range(min(qps, len(texts) - i))}
#         all_keys = list(responses.keys())

#         def response_callback(request_id, response, exception):
#             nonlocal responses
#             # responses[request_id] = (response, exception)
#             if exception is not None:
#                 responses[request_id] = None
#             else:
#                 responses[request_id] = (response, exception)

#         for uid, text in zip(list(all_keys),texts[i:i+qps]):
#             batch_request.add(_make_request(text, service), callback=response_callback, request_id=uid)
#         batch_request.execute()
        
#         all_outputs.extend(list(responses.values()))

#         if i != stop_lim:
#             end = time.time()
#             if (start - end) < 1:
#                 time.sleep(1 - (start - end))
    
#     return [getAttribProb("TOXICITY", j) if j else 0 for j in all_outputs + [None] * (new_len - old_len)]

#     # return [j[0]["attributeScores"]["TOXICITY"]["summaryScore"]["value"] for j in all_outputs + [0] * (new_len - old_len)]
    
# if __name__ == "__main__":
#     # Some examples of toxicity scores on text
#     temp = get_toxicity(["She's Mexican, she can be our maid.", "She's so smart, she can be our best friend.", "She's our president, she can be our president."] * 100)
#     print(temp)
#     get_toxicity(["I don't like short hair on girls.", "I don't like short skirts on girls."])
#     embed()
