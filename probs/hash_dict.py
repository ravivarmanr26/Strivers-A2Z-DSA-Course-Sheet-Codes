# Exercise: Hash Table
# nyc_weather.csv contains new york city weather for first few days in the month of January. Write a program that can answer following,
# What was the average temperature in first week of Jan
# What was the maximum temperature in first 10 days of Jan
# Figure out data structure that is best for this problem
#
# Solution
#
# nyc_weather.csv contains new york city weather for first few days in the month of January. Write a program that can answer following,
# What was the temperature on Jan 9?
# What was the temperature on Jan 4?
# Figure out data structure that is best for this problem
#
# Solution
#
# poem.txt Contains famous poem "Road not taken" by poet Robert Frost. You have to read this file in python and print every word and its count as show below. Think about the best data structure that you can use to solve this problem and figure out why you selected that specific data structure.
#  'diverged': 2,
#  'in': 3,
#  'I': 8
# Solution
#
# Implement hash table where collisions are
# handled using linear probing. We learnt about linear probing in the video tutorial. Take the hash table implementation that uses chaining and modify methods to use linear probing. Keep MAX size of arr in hashtable as 10.
# weather={}
# with open("nyc_weather.csv") as f:
#     for line in f:
#         tokens=line.split(',')
#         print(tokens)
#         date=tokens[0].strip()
#         temp=(tokens[1]).strip()
#         weather[date]=temp
#     print(weather)

# arr=[]
# with open("nyc_weather.csv") as f:
#     for line in f:
#         tokens=line.split(',')
#         temp = int(tokens[1].strip())
#         arr.append(temp)
#     print(arr)
#
# aver=sum(arr[:7])/len(arr[0:7])
# print(aver)
#
# # What was the maximum temperature in first 10 days of Jan
# print(max(arr))
#
# weather_dict = {}
#
# with open("nyc_weather.csv","r") as f:
#     for line in f:
#         tokens = line.split(',')
#         day = tokens[0]
#         try:
#             temperature = int(tokens[1])
#             weather_dict[day] = temperature
#         except:
#             print("Invalid temperature.Ignore the row")
# # What was the temperature on Jan 9
# print(weather_dict['Jan 9'])
# # What was the temperature on Jan 4
# print(weather_dict['Jan 4'])
# words={}
# c=0
# with open("poem.txt") as f:
#     for line in f:
#         tokens = line.split(' ')
#         # print(tokens)
#         for token in tokens:
#             token=token.replace('\n','')
#             # print(tokens)
#             if token in words:
#                 words[token] +=1
#             else:
#                 words[token] = 1
# print(words)

# def find_unique_elements(arr):
#     unique_elements = set(arr)
#     return list(unique_elements)
# arr=[1,2,3,5,6,777,45,12,1,2,3,4,5,5,5,3,4]
# print(find_unique_elements(arr))

# def find_duplicates(arr):
#     element_count = {}
#     duplicates = []
#     for element in arr:
#         if element in element_count:
#             duplicates.append(element)
#             print(duplicates)
#         else:
#             element_count[element] = 1
#             print(element_count)
#     return duplicates
# arr=[1,2,3,4,5,1,2,3]
# print(find_duplicates(arr))


# 2. Character Hashing
# Character hashing helps in various string manipulation tasks, such as finding the first non-repeating character in a string:
# def first_non_repeating_char(s):
#     char_count = {}
#     for char in s:
#         char_count[char] = char_count.get(char, 0) + 1
#     for char in s:
#         if char_count[char] == 1:
#             return char
#     return None
# s='ravivarman'
# print(first_non_repeating_char(s))

# def max_occurrence(arr):
#     element_count = {}
#     max_count = 0
#     max_element = None
#     for element in arr:
#         element_count[element] = element_count.get(element, 0) + 1
#         if element_count[element] > max_count:
#             max_count = element_count[element]
#             max_element = element
#     return max_element
# arr = [1, 3, 2, 1, 4, 1]
# print(max_occurrence(arr))

def hash_large_numbers(arr):
    large_hash = {}
    for number in arr:
        hashed_value = hash(number)
        large_hash[hashed_value] = number
    return large_hash
arr = [123456789, 987654321]
print(hash_large_numbers(arr))