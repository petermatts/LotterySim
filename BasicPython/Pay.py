rate = input("Enter hourly pay rate: ")
hours = input("Enter hours worked: ")
pay = float(rate) * float(hours)
print("Pay: $", pay)
if pay >= 100:
    print("Thats alot!");