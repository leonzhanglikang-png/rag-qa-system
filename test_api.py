import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("❌ 没读到API key,检查.env文件")
    exit()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好,请用一句话介绍你自己"}]
)
print("✅ API调用成功")
print("回复:", response.choices[0].message.content)
