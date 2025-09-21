from fastapi import FastAPI

app = FastAPI(title="Echo Backend")



@app.get("/")
def root():
    return {"message": "Echo backend is running"}
