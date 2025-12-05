arr = [1, 2, 3, 5, 4]
#traverse a array
# for index,num in enumerate(arr):
#     print(index,num)
##maximum element of the array
# max_ele=arr[0]
# for i in range(0,len(arr)+1):
#     if i>max_ele:
#         max_ele=i
# print(max_ele)
## sum of the array
# sum_of_arr=0
# for ele in arr:
#     sum_of_arr+=ele
# print(sum_of_arr)
##reverse the array
# left=0
# right = len(arr)-1
# while left < right:
#     arr[left],arr[right]=arr[right],arr[left]
#     left+=1
#     right-=1
# print(arr)
#Finding the Second Largest Element
# arr = [5,4,1,2,6]
# lar_num = 0
# second_lar = 0
# for num in arr:
#     if num > lar_num:
#         second_lar = lar_num
#         lar_num = num
#     elif num > second_lar and num != lar_num:
#         second_lar = num
#
# print("Largest number:", lar_num)
# print("Second largest number:", second_lar)
##Removing Duplicates from an Array
# arr = [1, 2, 2, 3, 4, 4, 5]
# #using set
# newarr=list(set(arr))
# print(newarr)
# #using a loop
# unique_element=[]
# for num in arr:
#     if num not in  unique_element:
#         unique_element.append(num)
# print(unique_element)
# Checking if Two Arrays are Equal
# arr1 = [1, 2, 3, 4, 5]
# arr2 = [1, 2, 3, 4, 5]
#
# print(arr1 == arr2)
# def are_arrays_equal(arr1, arr2):
#     return sorted(arr1) == sorted(arr2)
# arr1 = [1, 2, 3, 4, 5]
# arr2 = [1, 2, 3, 4, 5]
# print(are_arrays_equal(arr1,arr2))

#Rotating an Array to the Left by a Given Number of Positions
# arr = [1, 2, 3, 4, 5]
# k = 2
# n=len(arr)
# k=k%n
# print(arr[k:]+arr[:k])

##Finding the Missing Number in an Array of Consecutive Integers
# arr = [1, 2, 4, 5]
# n=len(arr)+1
# expected_sum=n*(n+1)//2
# sum_of_arr=sum(arr)
# print(expected_sum-sum_of_arr)

# Using XOR method
# missing_number = 0
# for i in range(1, n + 1):
#     missing_number ^= i
# for num in arr:
#     missing_number ^= num
# print(missing_number)

##Merging Two Sorted Arrays into One Sorted Array
# arr1 = [1, 3, 5]
# arr2 = [2, 4, 6]
# arr=arr1+arr2
# arr.sort()
# print(arr)
# arr1 = [1, 3, 5]
# arr2 = [2, 4, 6]
#
# merged_arr = []
# i = j = 0
#
# while i < len(arr1) and j < len(arr2):
#     if arr1[i] < arr2[j]:
#         merged_arr.append(arr1[i])
#         i += 1
#     else:
#         merged_arr.append(arr2[j])
#         j += 1
#
# # merged_arr.extend(arr1[i:])
# # merged_arr.extend(arr2[j:])
# print(merged_arr)
#Finding the Longest Increasing Subarray in an Array
# arr = [1, 2, 2, 3, 4, 1, 5, 6]
#
# max_len = curr_len = 1
# for i in range(1, len(arr)):
#     if arr[i] > arr[i - 1]:
#         curr_len += 1
#     else:
#         curr_len = 1
#     max_len = max(max_len, curr_len)
# print(max_len)
# def find_longest_increasing_subarray(arr):
#     if not arr:
#         return []
#     max_length = 1
#     current_length = 1
#     end_index = 0
#     for i in range(1, len(arr)):
#         if arr[i] > arr[i - 1]:
#             current_length += 1
#             if current_length > max_length:
#                 max_length = current_length
#                 end_index = i
#         else:
#             current_length = 1
#     return arr[end_index - max_length + 1:end_index + 1]
# arr = [1, 2, 2, 3, 4, 1, 5, 6]
# print(find_longest_increasing_subarray(arr))

# def find_unique_elements(arr):
#     counts = {}
#     unique_elements = []
#
#     # Step 1: Count the occurrences of each element
#     for i in range(len(arr)):
#         if arr[i] in counts:
#             counts[arr[i]] += 1
#         else:
#             counts[arr[i]] = 1
#     print(counts)
#     # Step 2: Collect elements that occur exactly once
#     for key in counts:
#         print("key",key,end=" ")
#         # print(counts[key])
#         if counts[key] == 1:
#             unique_elements.append(key)
#
#     return unique_elements
#
#
# # Example usage
# array = [1, 2, 3, 4, 5, 2, 3, 6, 7, 8, 9, 3]
# print(find_unique_elements(array))
array = [1, 2, 3, 4]


class Solution(object):
    def moveZeroes(self, nums):

        non_zero = 0  # Pointer for non-zero elements

        # Move all non-zero elements to the front
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[non_zero] = nums[non_zero], nums[i]
                non_zero += 1
