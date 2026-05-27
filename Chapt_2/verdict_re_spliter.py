import re
text = "Hello, world. Is this-- a test?."
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item for item in result if item.strip()]
print(result)