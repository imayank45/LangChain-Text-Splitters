from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

# Example usage
n = int(input("Enter a number: "))
if is_prime(n):
    print(f"{n} is a prime number.")
else:
    print(f"{n} is not a prime number.")

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=200,
    chunk_overlap=0,
    language=Language.PYTHON,
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[0])