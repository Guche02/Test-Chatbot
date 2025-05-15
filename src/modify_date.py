from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who converts abstract dates like 'next Monday', 'two weeks from today', etc., into a YYYY-MM-DD format. "
            "Today's date is {date} ({day_of_week}). Return only the final date in YYYY-MM-DD format, without extra explanation.",
        ),
        ("human", "{question}")
    ]
)

def convert_date(question: str) -> str:
    today = datetime.today()
    date_str = today.strftime("%Y-%m-%d")
    weekday = today.strftime("%A") 

    filled_prompt = prompt.invoke({
        "date": date_str,
        "day_of_week": weekday,
        "question": question
    })


    result = llm.invoke(filled_prompt)

    return result.content

if __name__ == "__main__":
    question = "next Monday"
    converted = convert_date(question)
    print(f"Input: {question}")
    print(f"Converted Date: {converted}")
