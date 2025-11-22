from backend.groq_client import get_groq_client
from backend.openai_client import get_openai_client

def get_chat_response(user_input):

    # Try Groq first
    try:
        groq_client = get_groq_client()
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a helpful crime analysis assistant."},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message["content"]

    except Exception as e:
        print("Groq failed:", e)

        # Fallback to OpenAI
        try:
            openai_client = get_openai_client()
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful crime analysis assistant."},
                    {"role": "user", "content": user_input}
                ]
            )
            return response.choices[0].message.content

        except Exception as e2:
            return f"AI Error: {str(e2)}"
