import datetime

class Logger:
    def __init__(self, exp_name):
        self.file = open('./logs/{}.log'.format(exp_name), 'w')

    def log(self, content):
        print(content)
        self.file.write(str(content) + '\n')
        self.file.flush()



