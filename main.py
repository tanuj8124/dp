from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
from sklearn.preprocessing import PolynomialFeatures
pr= PolynomialFeatures(4)



# Load the trained model
model = joblib.load("depth_predict.joblib")

# Define the input data structure
class InputData(BaseModel):
    param1: float
    param2: float
    param3: float

# Create FastAPI instance
app = FastAPI()

@app.get('/')
def index():
    return{'message':'api loaded'}

# Define the prediction endpoint
@app.post("/predict/")
async def predict(data: InputData):
    x_test_poly= pr.fit_transform([[data.param1, data.param2, data.param3]])
    prediction = model.predict([x_test_poly])
    return {"prediction": prediction[0]}

if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# To run the server, use the command:
# uvicorn api:app --reload
