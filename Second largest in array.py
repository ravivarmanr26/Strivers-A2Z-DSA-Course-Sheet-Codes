class Solution:
    def print2largest(self, arr, n):
        # Initialize the first and second largest variables
        first = second = float('-inf')

        # Traverse the array to find the largest and second largest elements
        for i in range(n):
            if arr[i] > first:
                second = first
                first = arr[i]
            elif arr[i] > second and arr[i] != first:
                second = arr[i]

        # If second largest is still negative infinity, it means no second largest found
        if second == float('-inf'):
            return -1
        else:
            return second

