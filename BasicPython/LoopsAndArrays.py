#Sums all numeric values in array/list
def sumArr(arr):
    sum = 0
    for i in arr:
        if type(i) is int or type(i) is float:
            sum += i
    return sum


list = [1, 2, 3, 4, 'hello']
print(sumArr(list))