import sys
from commands import commands

commands = commands()


def main():
    argc = len(sys.argv)
    command = "getTheJSON"
    
    if argc <= 1: # no parameter has been passed
        getattr(commands,command)()

if __name__ == "__main__":
    main()
