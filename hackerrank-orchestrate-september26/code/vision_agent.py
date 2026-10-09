from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

client = genai.Client()

def image_analyzer(img_path, event_type, description):
    #using Gemini's vision capabilities to extract required data
    try:
        img_file = Image.open(img_path)
        

        prompt = (
            f"Analyze the image and identify the financial amount corresponding to the transaction type '{event_type}' "
            f"(Context/Details: {description}).\n"
            "OUTPUT RULES:\n"
            "- Respond ONLY with the numerical value using a period (.) as the decimal separator.\n"
            "- Remove thousands separators (commas), currency symbols ($, €, etc.), and any text.\n"
            "- Output format example: 1500 or 1500.00\n"
            "- If NO relevant amount matches this transaction in the image, respond strictly with: NULL"
)
        
        model_ans = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[prompt, img_file]
        )
        
        clean_text = model_ans.text.strip().replace("$", "").replace(",", "")
        return float(clean_text)
        
    except Exception as e:
        print(f"[!] Unable to process img_file {img_path}: {e}")
        return 0.0