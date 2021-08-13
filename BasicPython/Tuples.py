#Immuatable list, cannot set/change values or sort

#Example assignments (left-hand side)
(x, y) = (0, 4)
#print(x) #0
#print(y) #4

#tuples are comparable
bool1 = (0, 1, 2) < (5, 1, 2)
print(bool1) #True
bool2 = ('Jones', 'Sally') > ('Jones', 'Same')
print(bool2) #False
