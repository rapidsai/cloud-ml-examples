import os
import json
import requests

host = 'localhost'
port = '56767'

headers = {
    "Content-Type": "application/json",
    "format": "pandas-split"
}

data = { 
    "columns": ["Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime", "CRSArrTime", "UniqueCarrier",
                "FlightNum", "ActualElapsedTime", "Origin", "Dest", "Distance", "Diverted"],
    "data": [[1987, 10, 1, 4, 1, 556, 0, 190, 247, 202, 162, 1846, 0]]
}

resp = requests.post(url="http://%s:%s/invocations" % (host, port), data=json.dumps(data), headers=headers)
print('Classification: %s' % ("ON-Time" if resp.text == "[0.0]" else "LATE"))
