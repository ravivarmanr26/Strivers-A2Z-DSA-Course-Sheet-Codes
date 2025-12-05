from typing import List  # Import List from typing

class Solution:
    def check(self, nums: List[int]) -> bool:
        # Find the pivot point where the order breaks
        pivot = -1
        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                print(nums[i])
                pivot = i
                print(pivot)
                break

        # If no pivot point is found, the list is already sorted
        if pivot == -1:
            return True

        # Check if the list can be sorted by rotating at the pivot point
        rotated_nums = nums[pivot + 1:] + nums[:pivot + 1]

        # Verify if the rotated list is sorted
        for i in range(len(rotated_nums) - 1):
            if rotated_nums[i] > rotated_nums[i + 1]:
                return False

        return True

# Example usage:
solution = Solution()
nums = [3, 4, 5, 1, 2]
print(solution.check(nums))  # Output: True
