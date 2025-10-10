***API Documentation***

In order for the API to work, you need to have the server running. You can start the server by executing:
```bash
python api.py
```

After starting the server, you can interact with the API using `curl` commands as shown below.


**Uploading data:**
To upload a JSON file to the API, you can use the following `curl` command:
```bash
curl -X POST "http://0.0.0.0:8000/upload_json" \
  -H "accept: application/json" \
  -F "file=@./test_data_for_api/test_data1.json"
```

**Sending single queries**
```bash
curl -X POST "http://0.0.0.0:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"text": "dog", "k": 3}'
```

**Sending multiple queries**
```bash
curl -X POST "http://0.0.0.0:8000/search_batch" \
  -H "accept: application/json" \
  -F "file=@./test_data_for_api/test_query1.jsonl"
```

```bash
curl -X POST "http://0.0.0.0:8000/search_batch" \
  -H "Content-Type: application/json" \
  -d '[{"text": "dog", "k": 3}, {"text": "cat", "k": 3}]'
```

**Deleting all data from index**
```bash
curl -X DELETE "http://0.0.0.0:8000/delete_all" \
  -H "accept: application/json"
```