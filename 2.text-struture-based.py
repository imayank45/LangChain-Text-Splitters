from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Consistency is one of the most underrated yet powerful traits a person can develop. It’s not about making huge leaps in a single day but about showing up repeatedly, even when motivation fades. Success, in any field—be it fitness, learning, or career—comes from small, consistent efforts that compound over time.

When you practice something regularly, your brain builds stronger neural connections, making the task feel easier and more natural. This principle applies to learning a language, mastering a skill, or developing discipline. For instance, studying for 30 minutes daily is far more effective than cramming for five hours once a week. The same goes for habits like reading, exercising, or even maintaining relationships—consistency nurtures growth and trust.

However, being consistent doesn’t mean being perfect. It means committing to progress despite setbacks. Some days you’ll move slowly, but as long as you don’t quit, you’re still moving forward. Over time, these small actions create a ripple effect that transforms not just your results but your mindset.

In the end, consistency turns ordinary efforts into extraordinary outcomes—and that’s what separates those who dream from those who achieve.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])