import json
import numpy as np
import pandas as pd
print(pd.DataFrame({}))


def lambda_handler(event, context):
    return {
        'statusCode': 200,
        'body': json.dumps('I guess it worked!')
    }

