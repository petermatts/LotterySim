#Accessing a element of a string
str = 'Oppa'
print(str[1]) #the first p

#Accessing range of elements
print(str[1:3]) #from 1 to 3 index (inclusive lower bound, exclusive upper bound)
print(str[:2]) #from beginning to index 2 (exclusive)
print(str[1:]) #from 1 to end of string
print(str[:]) #just the whole string

#Concatination
str1 = 'hello'
str2 = 'world'
print(str1 + ' ' + str2)

#Using in as a logical operator
fruit = 'banana'
print('n' in  fruit) #true
print('m' in fruit) #false

#String comparison
word = input('Enter word: ')
if word == fruit:
    print('All right, bananas.')
elif word < fruit:
    print('Your word,' + word + ', comes before banana')
elif word > fruit:
    print('Your word,' + word + ', comes after banana')
else:
    print('Oh Bananas... :/')

#SOME String methods
string = 'Hello there Python'
print(string.lower()) #hello there python
print(string.upper()) #HELLO THERE PYTHON
print(string.capitalize()) #Hello There Python
print(string.find('Py')) #
print(string.replace('Python', 'Javascript')) #Hello there Javascript
#And many many more...
