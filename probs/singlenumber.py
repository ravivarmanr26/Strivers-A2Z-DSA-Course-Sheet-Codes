nums = [2, 2, 1]  # Example list

single_number = 0
for i in nums:
    print(i)
    single_number ^= i

print(single_number)