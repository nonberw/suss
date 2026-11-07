"""List macros that are '??' in paper/numbers.tex and whether the paper text uses them (unresolved numbers in the PDF)."""
import glob
import re

nums = open("paper/numbers.tex", encoding="utf-8").read()
missing = re.findall(r"\\newcommand\{\\(\w+)\}\{\?\?\}", nums)
text = "".join(open(f, encoding="utf-8").read() for f in glob.glob("paper/*.tex") if not f.endswith("numbers.tex"))
used = [m for m in missing if re.search(r"\\" + m + r"(?![A-Za-z])", text)]
print(f"{len(missing)} missing macros; {len(used)} of them are used in the paper:")
print(" ".join(used))
pend = re.findall(r"\\PENDING\{(\w+)\}", text)
print("PENDING markers in text:", pend)
