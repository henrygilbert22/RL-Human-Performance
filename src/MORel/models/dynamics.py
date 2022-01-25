from enviroment import Enviroment

class Dynamics:
    
    look_behind: int
    env: Enviroment
    data: list
    
    X: list
    Y: list
    
    def __init__(self, look_behind: int) -> None:
        
        self.look_behind = look_behind
        self.env = Enviroment()
        
        self.data = self.env.get_dataset()
        
    def shift_data(self):
        
        for i in range(self.look_behind, len(self.data) - 1, 1):
            self.X.append(self.data[i-self.look_behind: self.look_behind])
            self.Y.append(self.data[i+1])
        
def main():
    d = Dynamics()

if __name__ == '__main__':
    main()