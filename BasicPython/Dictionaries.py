#The python equivalent of a Java Map or HashMap object
#?Also very similar to a basic JS object

#example
purse = dict()
purse['money'] = 12
purse['candy'] = 3
purse['tissues'] = 15
print(purse)
# print(purse['candy'])
purse['candy'] = purse['candy'] + 2


#another example
counts = dict()
names = ['bob', 'kalib', 'jill', 'mary', 'mary', 'bob', 'jill', 'kalib', 'jill']
# for name in names:
#     if name not in counts:
#         counts[name] = 1
#     else:
#         counts[name] += 1
#Does the same thing as the above commented code
for name in names:
    counts[name] = counts.get(name, 0) + 1
print(counts)

#Storing other collection (list) objects in dictionary
fruits = dict()
while True:
    fruit = input("Enter a fruit (or done to exit): ")
    if fruit == 'done' : break

    firstLetter = fruit[0].upper()
    fList = fruits.get(firstLetter, list())
    fList.append(fruit)
    fruits[firstLetter] = fList
print(fruits)
