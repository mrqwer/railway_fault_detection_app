from pydantic import BaseModel

class History(BaseModel):
    date: str
    image: bytes
    prediction: str

class Feedback(BaseModel):
    image: bytes
    prediction: str
    actual: str
