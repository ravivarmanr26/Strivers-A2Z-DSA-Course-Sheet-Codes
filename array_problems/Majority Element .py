class Solution(object):
    def majorityElement(self, nums):
        n = len(nums)
        threshold = n // 3

        # Dictionary to count occurrences of each element
        count_map = {}
        result = []

        # Count occurrences of each element
        for num in nums:
            if num in count_map:
                count_map[num] += 1
            else:
                count_map[num] = 1

        # Check which elements appear more than threshold times
        for num, count in count_map.items():
            if count > threshold:
                result.append(num)

        return result


# Example usage:
nums = [1, 3, 3, 2, 2, 4, 3, 3, 5, 5, 6, 3, 3, 7, 7]
sol = Solution()
print(sol.majorityElement(nums))