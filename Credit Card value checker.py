#Checks credit card value (runtime arg) for validity
import cs50
import sys
from sys import argv

#Input and filter
cc = cs50.get_int("Input CC Number: ")
if(cc<= 0):
    print("ERROR: INVALID cc number")
    sys.exit()

#Store digits into array
cc_digits = []

def buffer_adder(cc_no, digitarray):
    digitarray.append(cc_no%10)
    if(cc_no>9):
        buffer_adder(cc_no//10, digitarray)
    else:
        return()

buffer_adder(cc, cc_digits)

#Store n+2 and nth digits into respective arrays, add to checksum
checksum=0

for i in range(len(cc_digits)):
    #If index is odd
    if(i%2 != 0 and (cc_digits[i]*2)<=9):
        checksum += (cc_digits[i]*2)
    elif(i%2 !=0 and (cc_digits[i]*2)>9):
        checksum += (cc_digits[i]*2)%10
        checksum += ((cc_digits[i]*2)//10)%10
    #Even indeces
    else:
        checksum+= cc_digits[i]

def cc_IDer(cc_digits):
    #3 Credit card holder identifier
    if(cc_digits[15]==-1 and cc_digits[14]==3 and (cc_digits[13]==4 or cc_digits[13]==7)):
        print("AMEX")
    elif(cc_digits[15]==5 and (cc_digits[14]==1 or cc_digits[14]==2 or cc_digits[14]==3 or cc_digits[14]==4 or cc_digits[14]==5)):
        print("MASTERCARD")
    elif((cc_digits[15]==4 or (cc_digits[12]==4 and cc_digits[15]==-1 and cc_digits[14]==-1 and cc_digits[13]==-1))):
        print("VISA")
    else:
        print("INVALID")

if(checksum%10 == 0):
    cc_IDer(cc_digits)
else:
    print("ERROR: INVALID CARD")
