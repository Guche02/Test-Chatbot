from typing import Optional
from datetime import date
from pydantic import BaseModel, Field, EmailStr, model_validator
from src.chatbot.modify_date import convert_date  

def test():
    print("Test function in model.py")

class Person(BaseModel):
    """Information about a user."""
    name: Optional[str] = Field(default=None, description="The name of the user.")
    email: Optional[EmailStr] = Field(default=None, description="The email of the user.")
    phone: Optional[str] = Field(default=None, description="The phone number of the user.")   
    appointment_date: Optional[date] = Field(default=None, description="The date of the appointment booked by the user.")
    
    @model_validator(mode="before")  # runs before validation
    @classmethod                     # Required because no instance exists yet at validation time
    def normalize_appointment_date(cls, values: dict) -> dict:
        date_text = values.get("appointment_date")
        if date_text:
            try:
                converted = convert_date(date_text)
                values["appointment_date"] = converted
            except Exception as e:
                print(f"Warning: Failed to convert date '{date_text}': {e}")
        return values

if __name__ == "__main__":
    user = Person(
        name="Jaganath",
        email="jaganath@example.com",
        phone="9800000000",
        appointment_date="next Friday"
    )

    print(user.appointment_date)  
