"""
01 - Python strings, formatting, and counting values

Topics:
    1. Multi-line strings and indexing
    2. Looping through strings
    3. Checking membership (in / not in)
    4. Slicing
    5. String methods (upper, lower, replace, strip, split, ...)
    6. print() with end=
    7. f-strings and number formatting
    8. Escape characters and quotes
    9. Counting: str.count, find, and the same idea in other data types
   10. np.unique with return_counts (zip/dict, flattening, axis, string labels)
"""

import numpy as np

# ============================================================
# 1. Multi-line strings and indexing
# ============================================================
text = """Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua."""

print(text)
print(text[7])    # indexing starts at 0
print(text[5])

# ============================================================
# 2. Looping through strings
# ============================================================
word = 'Neuroscientist'

for char in word:
    print(char)

print(len(word))  # number of characters -> 14

# ============================================================
# 3. Checking membership: in / not in
# ============================================================
sentence = 'The best things in life are free!'

print("free" in sentence)       # True  (returns a boolean)
print('world' in sentence)      # False
print("expensive" not in sentence)  # True

if 'free' in sentence:
    print("YES!")
else:
    print("NO..")

if 'free' in sentence:
    print("Yes, 'free' is present.")

# ============================================================
# 4. Slicing  ->  [start:stop:step]
# ============================================================
# - start is included, stop is NEVER included
# - [:5]    first 5 characters
# - [5:]    from position 5 to the end
# - [2:6]   from position 2 up to (not including) 6
# - [-3:]   last 3 characters
# - [::2]   every 2nd character
# - [::-1]  reversed
greeting = "Hello, World!"

print(greeting[2:5])     # llo
print(greeting[:5])      # Hello
print(greeting[:-5])     # Hello, W
print(greeting[-3:])     # ld!
print(greeting[-5:])     # orld!
print(greeting[-5:-1])   # orld
print(greeting[::2])     # every 2nd character
print(greeting[::-1])    # reversed

# ============================================================
# 5. String methods (each returns a NEW string; original is unchanged)
# ============================================================
quote = " Don't just dream horizontally, DREAM VERTICALLY. "

print(quote.upper())
print(quote.lower())
print(quote.replace("horizontally", ""))    # removes the word
print(quote.replace("horizontally", " "))   # replaces it with a space
print(quote.strip())                        # removes leading/trailing whitespace
print(quote.split(','))                     # list; text between separators becomes the items

demo = "python is FUN!"
print(demo.capitalize())   # first char upper, the rest lower -> "Python is fun!"

# ============================================================
# 6. print() with end=
# ============================================================
# By default print() ends with a newline; end="" keeps the next print on the same line.
print("hi femy,", end="")
print("howzit going?")

print("hi femy,", end=""); print("howzit going?")   # ';' puts two statements on one line

# ============================================================
# 7. f-strings and number formatting
# ============================================================
# Use f-strings to combine strings and numbers.
age = 26
print(f"I am {age} years old")

price = 9
print(f'The price is {price:.2f} dollars')   # The price is 9.00 dollars
print(f'The price is {9:.2f} dollars')       # works on any expression, not just variables
# :.2f -> format as a float with 2 decimal places

# ============================================================
# 8. Escape characters and quotes
# ============================================================
print('why can\'t you say no?')     # \' = literal apostrophe
print("why can't you say no?")      # alternative: double quotes outside
print('''why can't you say no?''')  # alternative: triple quotes

# ============================================================
# 9. Counting: str.count and find (plus the same idea in other types)
# ============================================================
# count() needs an argument: what to count.
print(demo.count("!"))    # 1
print(demo.find('i'))     # index of first match -> 8
print(demo.find('k'))     # -1 means not found

# Counting cheat sheet:
#   NumPy array : np.unique(arr, return_counts=True)
#   DataFrame   : df['col'].value_counts()
#   String      : text.count('x')
#   List        : my_list.count('x')
#   Tuple       : my_tuple.count('x')
#   Dictionary  : list(d.values()).count('x')   # counts matching values

# ============================================================
# 10. np.unique with return_counts
# ============================================================

# --- 10.1 Basics ---
arr = np.array([3, 1, 2, 3, 3, 1])
values, counts = np.unique(arr, return_counts=True)
print(values)   # [1 2 3]  -> sorted unique values
print(counts)   # [2 1 3]  -> how many times each appears (same index order)

# --- 10.2 Why zip() is needed in dict(zip(values, counts)) ---
# dict() needs (key, value) pairs, not two separate arrays.
# zip() pairs items by position: [(1, 2), (2, 1), (3, 3)]
print(list(zip(values, counts)))
print(dict(zip(values, counts)))          # {1: 2, 2: 1, 3: 3}

# Equivalent loop version:
result = {}
for i in range(len(values)):
    result[values[i]] = counts[i]

# Keys/values are np.int64 by default; use .tolist() for plain Python ints
print(dict(zip(values.tolist(), counts.tolist())))

# Most frequent value
print(values[np.argmax(counts)])          # 3

# --- 10.3 Multi-dimensional arrays are flattened by default ---
arr2d = np.array([[1, 2, 2],
                  [3, 1, 2]])
values, counts = np.unique(arr2d, return_counts=True)
print(values)   # [1 2 3]
print(counts)   # [2 3 1]   -> treated as [1, 2, 2, 3, 1, 2]

# With axis=0 it finds unique ROWS (axis=1 -> unique columns)
arr_rows = np.array([[1, 2],
                     [3, 4],
                     [1, 2]])
rows, counts = np.unique(arr_rows, axis=0, return_counts=True)
print(rows)     # [[1 2]
                #  [3 4]]
print(counts)   # [2 1]

# --- 10.4 Works on strings: counting class labels (sleep stages) ---
labels = np.array(['W', 'N1', 'N2', 'N2', 'N3', 'REM', 'N2', 'W', 'REM'])
stages, counts = np.unique(labels, return_counts=True)
print(stages)   # ['N1' 'N2' 'N3' 'REM' 'W']  -> sorted alphabetically
print(counts)   # [1 3 1 2 2]
print(dict(zip(stages, counts)))

# Percentages -> quick check for class imbalance before training a model
percent = counts / counts.sum() * 100
print(percent.round(1))                   # [11.1 33.3 11.1 22.2 22.2]

# Custom (natural sleep-stage) order instead of alphabetical
order = ['W', 'N1', 'N2', 'N3', 'REM']
counts_dict = dict(zip(stages, counts))
print([counts_dict[s] for s in order])    # [2, 1, 3, 1, 2]
