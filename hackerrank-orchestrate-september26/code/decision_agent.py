from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

def inference_generation(user_text, json_context, format_answer):
    
    format_str = str(format_answer)

    prompt = (
        "You are an expert assistant in financial data analysis and request evaluation.\n\n"
        f"USER CONTEXT (JSON):\n{json_context}\n\n"
        f"USER COMMENT:\n{user_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the user comment and the provided JSON context to make your decision.\n"
        "2. Provide your response ONLY in English.\n"
        f"3. You MUST strictly follow this output structure/columns format:\n{format_str}\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT include markdown code blocks (e.g., no ```csv or ```).\n"
        "- Do NOT add introductory or conversational text.\n"
        "- Output ONLY the formatted raw data."
)


    response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            temperature=0.1,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text}
            ]
        )
    return response.choices[0].message.content.strip()