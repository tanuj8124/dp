
import uvicorn

from fastapi import FastAPI
import joblib 

from sklearn.preprocessing import PolynomialFeatures
import logging
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
pr= PolynomialFeatures(4)

# Load the trained model
model = joblib.load("dp.joblib")

# Initialize FastAPI
app = FastAPI()
origins = ["*"]  # Replace with specific origins for security in production

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],   

)


# Define a GET method that takes the three parameters as query parameters
@app.get("/predict/")



def predict(temperature: float, precipitation: float, month: int):
    try:
        logging.info(f"Received inputs - Temperature: {temperature}, Precipitation: {precipitation}, Month: {month}")
        
        poly= pr.fit_transform([[(temperature),(precipitation),(month)]])
        prediction = model.predict(poly)
        logging.info(f"Prediction result: {prediction}")
        return {"prediction": prediction[0]}
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run the server using: uvicorn filename:app --reload
