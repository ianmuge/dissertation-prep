import time

def log(message,state="INFO"):
    return print(time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()),":["+state+"]",message)