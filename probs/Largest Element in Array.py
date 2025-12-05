def largest(arr, n):
    c = 0
    for i in range(n):
        if arr[i] > c:
            c = arr[i]
    # print(arr)
    return c



