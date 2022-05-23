import sys

def getValue(name, age):
    names = (name + " : " + age)
    return names
    
if __name__ == "__main__":
    getValue(sys.argv[1], sys.argv[2])