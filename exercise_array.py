# Exercise: Array DataStructure
# Let us say your expense for every month are listed below,
# January - 2200
# February - 2350
# March - 2600
# April - 2130
# May - 2190
# Create a list to store these monthly expenses and using that find out,
#
# 1. In Feb, how many dollars you spent extra compare to January?
# 2. Find out your total expense in first quarter (first three months) of the year.
# 3. Find out if you spent exactly 2000 dollars in any month
# 4. June month just finished and your expense is 1980 dollar. Add this item to our monthly expense list
# 5. You returned an item that you bought in a month of April and
# got a refund of 200$. Make a correction to your monthly expense list
# based on this
#
# expense = [2200,2350,2600,2130,2190]
# print(" you spent extra compare to January",expense[1]-expense[0])
# s=0
# for i in range(len(expense)):
#     if expense[i]==2000:
#         s=1
#     else:
#         s=0
# if s==1:
#     print("yes")
# else:
#     print("no don't exactly 2000 dollars in any month")
# t=0
# for i in range(3):
#     t+=expense[i]
# print("your total expense in first quarter (first three months) of the year",t)
# expense.insert(6,1980)
# print(expense)
# expense[3]=expense[3]-200
# print(expense)


# You have a list of your favourite marvel super heros.
# heros=['spider man','thor','hulk','iron man','captain america']
# Using this find out,
#
# 1. Length of the list
# 2. Add 'black panther' at the end of this list
# 3. You realize that you need to add 'black panther' after 'hulk',
#    so remove it from the list first and then add it after 'hulk'
# 4. Now you don't like thor and hulk because they get angry easily :)
#    So you want to remove thor and hulk from list and replace them with doctor strange (because he is cool).
#    Do that with one line of code.
# 5. Sort the heros list in alphabetical order (Hint. Use dir() functions to list down all functions available in list)

# heros=['spider man','thor','hulk','iron man','captain america']
#
# print(len(heros))
# heros.append('black panther')
# print(heros)
# heros.remove('black panther')
# heros.insert(3,'black panter')
# print(heros)
# # Print the list of attributes and methods of the list class
# print(dir(list))
#
# heros.remove('hulk')
# heros.remove('thor')
# print(heros)
# heros.insert(1,'docter strange')
#
# print(heros)
# print(sorted(heros))

# Create a list of all odd numbers between 1 and a max number. Max number is something you need to
# take from a user using input() function
def odd(n):
    l=[]
    for i in range(1,n):
        if i%2!=0:
            l.append(i)
    return l
s=int(input())
print(odd(s))
