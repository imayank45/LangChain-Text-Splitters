from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

text = """
During the 1980s and early 1990s, Punjab faced a period of intense terrorism driven by the Khalistan movement, which sought to create a separate Sikh homeland. The movement began as a political demand but soon escalated into armed conflict, leading to widespread violence and instability in the region. Militants targeted government officials, civilians, and infrastructure, resulting in thousands of deaths and a climate of fear. The Indian government responded with strict counter-terrorism measures, including military operations and police crackdowns, which eventually subdued the insurgency. While peace has largely returned to Punjab, the era remains a painful chapter in the state’s history, serving as a reminder of how political, religious, and social tensions can spiral into devastating conflict.
The farmer protest that began in 2020 saw massive participation from Punjab, as farmers united against three agricultural laws passed by the Indian government. They feared these laws would weaken the minimum support price (MSP) system and leave them vulnerable to corporate exploitation. Punjab’s farmers, known for their vital contribution to India’s food supply, led the movement with determination and solidarity, camping for months at Delhi’s borders. Their peaceful yet powerful protests drew national and international attention, eventually leading to the repeal of the laws in 2021. The movement highlighted the farmers’ resilience and the importance of dialogue between policymakers and the agricultural community.
"""

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

docs = text_splitter.create_documents([text])
print(len(docs))
print(docs)