import re

pattern1 = r"(--|[,.?_:;\"'()!]|\s)"
pattern2 = r"(--|[^0-9a-zA-Z])"
pattern3 = r"(--|[^0-9a-zA-Z]|\d+)"
text = "abc1-def2 -- ghi123. jkl 2 30 mno."
result1 = re.split(pattern1, text)
result2 = re.split(pattern2, text)
result3 = re.split(pattern3, text)
print(result1)
print(result2)
print(result3)