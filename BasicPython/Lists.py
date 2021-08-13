#constant list
l1 = [1,2,3,4,5]
l1.append(6) #append an element to end of list
# for l in l1:
    # print(l)

#list object
stuff = list()
stuff.append('book') #insert element
stuff.append('cookie')

stuff.sort() #sorts lists

#Cool functions
nums = l1
print(nums)
print('Length: ' + str(len(nums)))
print('Min: ' + str(min(nums)))
print('Max: ' + str(max(nums)))
print('Sum: ' + str(sum(nums)))
print('Avg: ' + str(sum(nums)/float(len(nums))))